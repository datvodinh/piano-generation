import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import TransfoXLConfig, TransfoXLModel, GPT2Model, GPT2Config
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import OneCycleLR



class MusicGenerativeModel(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 d_model         = 512,
                 nhead           = 8,
                 nlayers         = 6,
                 dropout         = 0.1,
                 dim_feedforward = 2048,
                 batch_first     = True,
                 lr              = 1e-4,
                 max_lr          = 3e-4,
                 beta            = (0.9,0.999),
                 total_steps     = 500_000,
                 pct_start       = 0.1,
                 model_type      = 'transformer'):
        
        if model_type == 'lstm':
            self.embed   = nn.Embedding(vocab_size,d_model)
            self.model   = nn.LSTM(input_size = d_model,
                                  hidden_size = d_model,
                                  num_layers  = nlayers,
                                  batch_first = batch_first)
            self.linear  = nn.Linear(d_model,vocab_size)
            self.forward = self._forward_lstm

        elif model_type == 'transformer':
            self.embed = nn.Embedding(vocab_size,d_model)
            self.position_embed = PositionalEncoding(num_hiddens = d_model,
                                                     dropout     = dropout,
                                                     max_len     = 10000)        
            layer = nn.TransformerDecoderLayer(d_model           = d_model,
                                                nhead            = nhead,
                                                dropout          = dropout,
                                                dim_feedforward  = dim_feedforward,
                                                batch_first      = batch_first)  
            self.model   = nn.TransformerDecoder(layer, num_layers = nlayers)
            self.fc      = nn.Linear(d_model,vocab_size)
            self.forward = self._forward_transformer
        
        elif model_type == 'transformer_xl':
            config = TransfoXLConfig(vocab_size = vocab_size,
                                 d_model    = d_model,
                                 n_head     = nhead,
                                 d_head     = d_model // nhead,
                                 d_inner    = dim_feedforward, 
                                 n_layer    = nlayers, 
                                 dropout    = dropout)
        
            self.model        = TransfoXLModel(config)
            self.fc           = nn.Linear(d_model,vocab_size)
            self.batch_first  = batch_first
            self.forward      = self._forward_transformer_xl

        elif model_type == 'gpt2':
            gpt2_config = GPT2Config(vocab_size = vocab_size,
                                    n_embd      = d_model,
                                    n_layer     = nlayers,
                                    n_head      = nhead,
                                    n_inner     = dim_feedforward,
                                    attn_pdrop  = dropout)
        
            self.model        = GPT2Model(gpt2_config)
            self.fc           = nn.Linear(d_model,vocab_size)
            self.batch_first  = batch_first
            self.forward      = self._forward_transformer_xl

        self.optim_config = {
            'lr':lr,
            'betas':beta,
            'max_lr':max_lr,
            'total_steps':total_steps,
            'pct_start':pct_start
            }


    def training_step(self,batch,batch_idx):
        src,tgt = batch
        outputs = self.forward(src)
        loss    = F.nll_loss(outputs,tgt)
        acc     = (outputs.view(-1,outputs.shape[-1]).argmax(dim=1) == tgt).float().mean()
        self.log('train_loss',loss.detach())
        self.log('train_accuracy',acc)
        return loss
    
    def configure_optimizers(self):
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
    
    def _forward_lstm(self,x):
        x   = self.embed(x) # B S E
        x,_ = self.model(x) # B S E
        x   = self.linear(x) # B S V
        return x

    def _forward_transformer(self,x):
        x = self.embed(x) * math.sqrt(self.d_model) # B S E
        x = self.position_embed(x) # B S E
        x = self.model(x,x,tgt_mask=self._target_mask(x)) # B S E
        x = self.fc(x) # B S V
        return x['last_hidden_state']
    
    def _forward_transformer_xl(self,x):
        if self.batch_first:
            x = x.transpose(0,1) # B S -> S B
        x = self.model(x) # S B E
        x = self.fc(x) # S B V
        return x['last_hidden_state'].transpose(0,1) # S B V -> B S V
    
    def _target_mask(self,target):
        mask = (torch.triu(torch.ones(target.shape[1], target.shape[1])) == 0).transpose(0, 1) # S S
        return mask.bool().to(self.device)

class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout = 0.2,max_len=1000):
        super().__init__()
        PE           = torch.zeros((1,max_len,num_hiddens))
        self.dropout = nn.Dropout(dropout)
        position     = torch.arange(0,max_len,dtype=torch.float32).reshape(-1,1) \
        / torch.pow(10000,torch.arange(0,num_hiddens,2,dtype=torch.float32) / num_hiddens)
        PE[:,:,0::2] = torch.sin(position)
        PE[:,:,1::2] = torch.cos(position)
        self.register_buffer('PE',PE)

    def forward(self,x):
        x = x + self.PE[:,:x.shape[1],:].to(x.device) # B S E
        return self.dropout(x)