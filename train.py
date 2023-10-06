from src.model.model import *
from src.utils.dataloader import MusicDataset, CollateFn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
import argparse
import pytorch_lightning as pl
import yaml
import wandb

# PARSER
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir','-d',type=str,
                    help='data directory',required=True)
parser.add_argument('--model_type','-mt',type=str,default='gpt2',
                    help='model type')
parser.add_argument('--wandb','-w',type=bool,default=False,
                    help='use wandb or not')
parser.add_argument('--max-epochs','-me',type=int,default=100,
                    help='max epoch')
parser.add_argument('--batch-size','-b',type=int,default=32,
                    help='batch size')
parser.add_argument('--lr','-l',type=float,default=1e-4,
                    help='learning rate')
parser.add_argument('--num-workers','-n',type=int,default=0,
                    help='number of workers')

args = parser.parse_args()

# MODEL
with open(f'./config/{args.model_type}.yaml') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)
config['model_type'] = args.model_type
model = MusicGenerativeModel(**config)

# DATALOADER
loader = DataLoader(dataset = MusicDataset(args.data_dir),
                    batch_size = args.batch_size,
                    collate_fn = CollateFn(),
                    num_workers = args.num_workers)

# WANDB
wandb.login(key = "844fc0e4bcb3ee33a64c04b9ba845966de80180e")
wandb.init(project = "music-generation",
           config = config,
           name = f"{config.model_type}",)
logger  = WandbLogger(project="music-generation",
                          log_model="all") \
            if args.wandb else None

# CALLBACK
ckpt_callback = ModelCheckpoint(
        monitor = "accuracy",
        dirpath = f"{config.save_path}/version_{config.version}",
        filename = "checkpoints-{epoch:02d}-{accuracy:.5f}",
        save_top_k = 3,
        mode = "max",
    )
lr_callback = LearningRateMonitor(logging_interval='step')

# TRAINER
def main():
    trainer = pl.Trainer(logger=logger,
                         callbacks=[ckpt_callback,lr_callback],
                         gradient_clip_val=1.0,
                         max_epochs=args.max_epochs,)
    trainer.fit(model,loader)

if __name__ == '__main__':
    main()

    