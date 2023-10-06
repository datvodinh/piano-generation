from typing import Any, Dict, List, Literal, Mapping, Optional, Union
import os
import sys
sys.path.append(os.getcwd())
import argparse
import torch
import torch.nn.functional as F
from random import randint, sample
from src.utils.vocab import *
from src.utils.tokenizer import *
tok = Tokenizer()

def processing_data(fname: str):
    
    encode_tensor, _ = tok.midi2tensor(fname)
    aug_data = _split_and_aug_data([encode_tensor], lth=512,factor=4)
    return aug_data,fname

def _split_and_aug_data(data,
                        lth: Optional[int] = 512,
                        factor: Optional[int] = 4):
    '''
    data: [tensor1,tensor2,...]
    '''
    list_aug_seq = []
    for d in data:
        for i in range(0,len(d),lth):
            start   = max(0,i-randint(0,lth//factor))
            end     = min(len(d),i+lth)
            seq     = d[start:end].view(1,-1)
            aug_seq = _aug(seq)
            list_aug_seq += aug_seq

    return list_aug_seq

def _aug(data:list):
    
    # SHIFT NOTE, OFFSET BY -2 TO 2 NOTE
    note_shifts    = [-2, -1, 0, 1, 2]
    note_shifted_data = []
    for seq in data:
        for shift in note_shifts:
            _shift = shift
            note_shifted_seq = []
            for idx in seq:
                _idx = idx + _shift
                if (0 < idx <= NOTE_ON and 0 < _idx <= NOTE_ON) or \
                        (NOTE_ON < idx <= NOTE_EVENTS and NOTE_ON < _idx <= NOTE_EVENTS):
                    note_shifted_seq.append(_idx)
                else:
                    note_shifted_seq.append(idx)
            note_shifted_seq = torch.LongTensor(note_shifted_seq)
            note_shifted_data.append(note_shifted_seq)

    # TIME STRETCH, STRETCH BY 0.9 TO 1.1 TIME
    time_stretches = [1 /1.1, 1/1.05, 1, 1.05, 1.1]
    time_stretched_data = []
    delta_time = 0
    for seq in note_shifted_data:
        for time_stretch in time_stretches:
            time_stretched_seq = []
            for idx in seq:
                if NOTE_EVENTS < idx <= NOTE_EVENTS + TIME_SHIFT:
                    time = idx - (NOTE_EVENTS - 1)
                    delta_time += torch.round(time * DIV * time_stretch+1e-4).int().item()
                else:
                    time_to_events(delta_time, index_list=time_stretched_seq)
                    delta_time = 0
                    time_stretched_seq.append(idx)

            time_stretched_seq = torch.LongTensor(time_stretched_seq)
            time_stretched_data.append(time_stretched_seq)

    aug_data = []
    for seq in time_stretched_data:
        aug_data.append(F.pad(F.pad(seq, (1, 0), value=start_token), (0, 1), value=end_token)) # ADD <SOS> AND <EOS>
    return aug_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir','-d',type=str,
                        help='data directory',required=True)
    parser.add_argument('--save_dir','-s',type=str,
                        help='save directory',required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for fname in os.listdir(data_dir):
        aug_data,fname = processing_data(os.path.join(data_dir,fname))
        for i,data in enumerate(aug_data):
            torch.save(data,os.path.join(save_dir,f'{fname}_{i}.pt'))