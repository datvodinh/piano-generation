import os
import sys
sys.path.append(os.getcwd())
import time
import argparse
import torch
import torch.nn.functional as F
from random import randint, sample
from src.utils.vocab import *
from src.utils.tokenizer import *
from src.utils.pbar import ProgressBar
sys.path.append(os.getcwd())
from concurrent.futures import ProcessPoolExecutor, as_completed

tok = Tokenizer()

def processing_data(fname: str):
    try:
        encode_tensor, _ = tok.midi2tensor(fname)
    except:
        return []
    aug_data = _split_and_aug_data([encode_tensor], lth=512,factor=8)
    return aug_data

def _split_and_aug_data(data,
                        lth: int = 512,
                        factor: int = 4):
    '''
    data: [tensor1,tensor2,...]
    '''
    list_aug_seq = []
    for d in data:
            seq_start = d[0:lth]
            seq_end   = d[-lth:]
            # start   = max(0,i-randint(0,lth//factor))
            # end     = min(len(d),i+lth)
            # seq     = d[start:end].view(1,-1)
            aug_seq = _aug([seq_start,seq_end])
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
    parser.add_argument('--num_workers','-n',type=int,default=8,
                        help='number of workers')
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.getcwd() not in data_dir:
        fdir = os.walk(os.path.join(os.getcwd(),data_dir))
    else:
        fdir = os.walk(data_dir)

    if os.getcwd() not in save_dir:
        save_dir = os.path.join(os.getcwd(),save_dir)

    list_dir = []
    for path, dirs, files in fdir:
        for name in files:
            f = os.path.join(path, name)
            if f.split(".")[-1] in ["mid", "midi"]:
                list_dir.append(f)
    print(f'TOTAL MID FILE: {len(list_dir)}')
    print(f'PROCESSING...')
    bar = ProgressBar(len(list_dir))
    idx = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for fname in list_dir:
            future = executor.submit(processing_data, fname)
            futures.append(future)
        bar.step(idx)
        data = []
        for future in futures:
            idx += 1
            data += future.result()
            bar.step(idx)
        
        data = torch.nn.utils.rnn.pad_sequence(data,batch_first=True,padding_value=pad_token)
        print(f"DATA SHAPE: {data.shape}")
        torch.save(data,os.path.join(save_dir,f"aug_data.pt"))
        print(f"SAVE AUG DATA TO {os.path.join(save_dir,f'aug_data.pt')}")