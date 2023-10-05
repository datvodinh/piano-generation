from pytorch_lightning.loggers import WandbLogger
from src.model.model import *
from src.utils.dataloader import LSTMLoader,TransformerLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import argparse
import pytorch_lightning as pl
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir','-d',type=str,help='data directory',required=True)
parser.add_argument('--model_type','-mt',type=str,default='gpt2',help='model type')
parser.add_argument('--wandb','-w',type=bool,default=False,help='use wandb or not')
parser.add_argument('--max-epochs','-me',type=int,default=100,help='max epoch')
parser.add_argument('--batch-size','-b',type=int,default=32,help='batch size')
parser.add_argument('--lr','-l',type=float,default=1e-4,help='learning rate')
parser.add_argument('--num-workers','-n',type=int,default=0,help='number of workers')

args = parser.parse_args()

with open(f'./config/{args.model_type}.yaml') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)
config['model_type'] = args.model_type
model = MusicGenerativeModel(**config)

if args.model_type == 'lstm':
    dataloader = LSTMLoader()
else:
    dataloader = TransformerLoader()

def main():
    ckpt_callback = ModelCheckpoint(
        monitor="accuracy",
        dirpath=f"{config.save_path}/version_{config.version}",
        filename="checkpoints-{epoch:02d}-{accuracy:.5f}",
        save_top_k=3,
        mode="max",
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    logger  = WandbLogger(project="music-generation",
                          log_model="all") \
            if args.wandb else None
    
    trainer = pl.Trainer(logger=logger,
                         callbacks=[ckpt_callback,lr_callback],
                         gradient_clip_val=1.0,
                         max_epochs=args.max_epochs,)
    trainer.fit(model)

if __name__ == '__main__':
    main()