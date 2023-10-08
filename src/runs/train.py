import os
import sys
sys.path.append(os.getcwd())
from src.model.model import *
from src.utils.dataloader import MusicDataset, CollateFn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import pytorch_lightning as pl
import yaml
import wandb

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def main():
    # PARSER
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir','-td',type=str,
                        help='training data directory',required=True)
    parser.add_argument('--val-dir','-vd',type=str,
                        help='validation data directory',required=True)
    parser.add_argument('--model-type','-mt',type=str,default='gpt2',
                        help='model type')
    parser.add_argument('--wandb','-w',default=False,action='store_true',
                        help='use wandb or not')
    parser.add_argument('--max-epochs','-me',type=int,default=100,
                        help='max epoch')
    parser.add_argument('--batch-size','-b',type=int,default=32,
                        help='batch size')
    parser.add_argument('--lr','-l',type=float,default=1e-4,
                        help='learning rate')
    parser.add_argument('--num-workers','-n',type=int,default=0,
                        help='number of workers')
    parser.add_argument('--seed','-s',type=int,default=42,
                        help='seed')
    parser.add_argument('--batch-chunk','-bc',type=int,default=1,
                        help='number of chunks in one mini-batch')

    args = parser.parse_args()

    # SEED
    seed_everything(args.seed)

    # DATALOADER
    train_loader = DataLoader(dataset = MusicDataset(args.train_dir),
                        batch_size = args.batch_size,
                        collate_fn = CollateFn(),
                        num_workers = args.num_workers,
                        shuffle = True)
    
    val_loader = DataLoader(dataset = MusicDataset(args.val_dir),
                        batch_size = args.batch_size,
                        collate_fn = CollateFn(),
                        num_workers = 0,
                        shuffle = False)
    # MODEL
    with open(os.path.join(os.getcwd(),"config",f"{args.model_type}.yaml")) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    config['model_type']  = args.model_type
    config['total_steps'] = len(train_loader) * args.max_epochs
    model = MusicGenerativeModel(**config)


    # WANDB
    if args.wandb:
        wandb.login(key = "844fc0e4bcb3ee33a64c04b9ba845966de80180e") # API KEY
        logger  = WandbLogger(project="music-generation",
                            name=f"{args.model_type}-{args.max_epochs}-{args.batch_size}",
                            log_model="all")
    else:
        logger = None

    # CALLBACK
    root_path = os.path.join(os.getcwd(),"checkpoints")
    ckpt_path = os.path.join(os.path.join(root_path,"ckpt/"))
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ckpt_callback = ModelCheckpoint(
            monitor = "val_acc_epoch",
            dirpath = ckpt_path,
            filename = "checkpoints-{epoch:02d}-{val_acc_epoch:.5f}",
            save_top_k = 3,
            mode = "max"
    )

    # TRAINER
    trainer = pl.Trainer(default_root_dir=root_path,
                         logger = logger,
                         callbacks = [ckpt_callback],
                         gradient_clip_val = 1.0,
                         max_epochs = args.max_epochs)
    trainer.fit(model,train_loader,val_loader)

if __name__ == '__main__':
    main()


    