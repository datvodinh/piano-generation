import os
import sys
sys.path.append(os.getcwd())
from src.model.model import *
from src.utils.tokenizer import *
from src.utils.vocab import *
import argparse
import torch
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint','-mt',type=str,required=True,
                        help='load model checkpoint')
    parser.add_argument('--max-length','-ml',type=int,default=1024,
                        help='max length of sequence generated')
    parser.add_argument('--temperature','-t',type=float,default=0.7,
                        help='temperature of sampling')
    parser.add_argument('--top-k','-k',type=int,default=40,
                        help='top k of sampling')
    parser.add_argument('--top-p','-p',type=float,default=0.9,
                        help='top p of sampling')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(os.getcwd(),"config",f"gpt2.yaml")) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    model = MusicGenerativeModel(**config)
    model.load_from_checkpoint(args.checkpoint,**config).to(device)
    
if __name__ == '__main__':
    main()
    
