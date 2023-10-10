import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import TransfoXLConfig, TransfoXLModel, GPT2Model, GPT2Config
from typing import Any, Optional,Dict
import math
from torch.optim.lr_scheduler import OneCycleLR
import os
import sys
sys.path.append(os.getcwd())
class MusicGenerativeModel(pl.LightningModule):
    def __init__(self,
                 vocab_size           = 416,
                 d_model: int         = 512,
                 nhead: int           = 8,
                 nlayers: int         = 6,
                 dropout: int         = 0.1,
                 dim_feedforward: int = 2048,
                 batch_first: int     = True,
                 lr: float            = 1e-4,
                 max_lr: float        = 3e-4,
                 beta: tuple          = (0.9,0.999),
                 total_steps: int     = 500_000,
                 pct_start: float     = 0.1,
                 mem_len: int         = 512,
                 model_type: str      = 'transformer'):
        super().__init__()
        
        if model_type == 'lstm':
            self.embed   = nn.Embedding(vocab_size,d_model)
            self.model   = nn.LSTM(input_size = d_model,
                                  hidden_size = d_model,
                                  num_layers  = nlayers,
                                  batch_first = batch_first)
            self.forward = self._forward_lstm

        elif model_type == 'transformer':
            self.embed = nn.Embedding(vocab_size,d_model,padding_idx=0)
            self.position_embed = PositionalEncoding(num_hiddens = d_model,
                                                     dropout     = dropout,
                                                     max_len     = 20000)        
            layer = nn.TransformerDecoderLayer(d_model           = d_model,
                                                nhead            = nhead,
                                                dropout          = dropout,
                                                dim_feedforward  = dim_feedforward,
                                                batch_first      = batch_first)  
            self.model   = nn.TransformerDecoder(layer, num_layers = nlayers)
            self.forward = self._forward_transformer
            self.sqrt_d_model = math.sqrt(d_model)
        
        elif model_type == 'transformer_xl':
            config = TransfoXLConfig(vocab_size = vocab_size,
                                     d_model    = d_model,
                                     d_embed    = d_model,
                                     mem_len    = mem_len,
                                     n_head     = nhead,
                                     d_head     = d_model // nhead,
                                     d_inner    = dim_feedforward, 
                                     n_layer    = nlayers, 
                                     dropout    = dropout,
                                     cutoffs    = [])
        
            self.model        = TransfoXLModel(config)
            self.batch_first  = batch_first
            self.forward      = self._forward_transformer_xl

        elif model_type == 'gpt2':
            gpt2_config = GPT2Config(vocab_size = vocab_size,
                                    n_positions = 2048,
                                    n_embd      = d_model,
                                    n_layer     = nlayers,
                                    n_head      = nhead,
                                    n_inner     = dim_feedforward,
                                    attn_pdrop  = dropout,
                                    bos_token_id= 414,
                                    eos_token_id= 415)
        
            self.model        = GPT2Model(gpt2_config)
            self.batch_first  = batch_first
            self.forward      = self._forward_transformer_xl

        self.fc = nn.Linear(d_model,vocab_size,bias=False)

        self.optim_config = {
            'lr':lr,
            'betas':beta,
            'max_lr':max_lr,
            'total_steps':total_steps,
            'pct_start':pct_start
            }
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)
        self.val_acc = []


    def training_step(self,batch,batch_idx:Optional[int] = None):
        src,tgt = batch # B S
        outputs = self.forward(src) # B S V
        outputs = outputs.reshape(-1,outputs.shape[-1]) # B*S V
        tgt     = tgt.reshape(-1) # B*S
        loss    = self.criterion(outputs,tgt)
        acc     = (outputs.argmax(dim=-1) == tgt).float().mean()
        try:
            self.log('train_loss',loss)
            self.log('train_accuracy',acc)
        except:
            pass

        return loss
    
    def validation_step(self,batch,batch_idx:Optional[int] = None):
        src,tgt = batch # B S
        outputs = self.forward(src) # B S V
        outputs = outputs.reshape(-1,outputs.shape[-1]) # B*S V
        tgt     = tgt.reshape(-1) # B*S
        loss    = self.criterion(outputs,tgt)
        acc     = (outputs.argmax(dim=-1) == tgt).float().mean()
        try:
            self.log('val_loss',loss)
            self.log('val_accuracy',acc)
            self.val_acc.append(acc.item())
        except:
            pass
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val_acc_epoch',sum(self.val_acc)/len(self.val_acc))
        
    
    def configure_optimizers(self)-> Dict[str, Any]:
        optim = torch.optim.AdamW(params = self.parameters(),
                                  lr     = self.optim_config['lr'],
                                  betas  = self.optim_config['betas'])
        sched = OneCycleLR(optimizer   = optim,
                           max_lr      = self.optim_config['max_lr'],
                           total_steps = self.optim_config['total_steps'],
                           pct_start   = self.optim_config['pct_start'])
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1
            }
        }
    
    def _forward_lstm(self,x:torch.Tensor):
        x   = self.embed(x) # B S E
        x,_ = self.model(x) # B S E
        x   = self.fc(x)    # B S V
        return x

    def _forward_transformer(self,x:torch.Tensor):
        x = self.embed(x) * self.sqrt_d_model             # B S E
        x = self.position_embed(x)                        # B S E
        x = self.model(x,x,tgt_mask=self._target_mask(x)) # B S E  
        x = self.fc(x)                                    # B S V
        return x
    
    def _forward_transformer_xl(self,x:torch.Tensor):            
        x = self.model(x)['last_hidden_state']       # B S E
        x = self.fc(x)                               # B S V    
        return x
    
    def _target_mask(self,target):
        mask = (torch.triu(torch.ones(target.shape[1], target.shape[1])) == 0).transpose(0, 1) # S S
        return mask.bool().to(self.device)
    
    @torch.no_grad()
    def generate(self,src,max_token=256,block_size=512,temperature=0.7):
        self.eval()
        for _ in range(max_token):
            outputs = self.forward(src[:,-block_size:])
            outputs = outputs[:,-1,:]
            outputs = outputs / temperature
            outputs = torch.softmax(outputs,dim=-1)
            src_out = torch.multinomial(outputs,num_samples=1)
            src     = torch.cat([src,src_out],dim=-1)
        return src
class PositionalEncoding(nn.Module):
    def __init__(self,
                 num_hiddens:int,
                 dropout: Optional[float] = 0.2,
                 max_len: Optional[int]   = 1000):
        super().__init__()
        PE           = torch.zeros((1,max_len,num_hiddens))
        self.dropout = nn.Dropout(dropout)
        position     = torch.arange(0,max_len,dtype=torch.float32).reshape(-1,1) \
        / torch.pow(10000,torch.arange(0,num_hiddens,2,dtype=torch.float32) / num_hiddens)
        PE[:,:,0::2] = torch.sin(position)
        PE[:,:,1::2] = torch.cos(position)
        self.register_buffer('PE',PE)

    def forward(self,x:torch.Tensor):
        x = x + self.PE[:,:x.shape[1],:].to(x.device) # B S E
        return self.dropout(x)