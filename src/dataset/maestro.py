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
import json

tok = Tokenizer()

def multi_processing_data(list_data,data_dir,num_workers: int = 8,lth:int = 512,training: bool = True):
    bar = ProgressBar(len(list_data))
    idx = 0
    if training:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for fname in list_data:
                f = os.path.join(data_dir,fname)
                future = executor.submit(processing_data, f, lth,8,training)
                futures.append(future)
            data = []
            for future in futures:
                idx += 1
                bar.step(idx)
                data+=future.result()
    else:
        data = []
        for fname in list_data:
            f = os.path.join(data_dir,fname)
            data+=processing_data(f,lth,8,training)
            idx += 1
            bar.step(idx)
    return torch.nn.utils.rnn.pad_sequence(data,batch_first=True,padding_value=0)

def processing_data(fname: str,lth: int = 512,factor: int = 4, training: bool = True):
    try:
        encode_tensor, _ = tok.midi2tensor(fname)
    except:
        return []
    if training:
        aug_data = _split_and_aug_data([encode_tensor], lth=lth,factor=factor)
        return aug_data
    else:
        try:
            return [encode_tensor[:lth]]
        except:
            return []

def _split_and_aug_data(data,
                        lth: int = 512,
                        factor: int = 8):
    '''
    data: [tensor1,tensor2,...]
    '''
    list_aug_seq = []
    for d in data:
            seq_start = d[0:min(len(d),lth + lth //8)]
            seq_end   = d[-min(len(d),lth + lth //8):]
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
    parser.add_argument('--lth','-l',type=int,default=512,
                        help='length of sequence to cut')
    parser.add_argument('--num_workers','-n',type=int,default=8,
                        help='number of workers')
    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(data_dir,"maestro-v3.0.0.json"), "r") as f:
        data = json.load(f)

    list_train = []
    list_val = []
    list_test = []

    for k in data['midi_filename'].keys():
        if data['split'][k] == 'train':
            list_train.append(data['midi_filename'][k])
        elif data['split'][k] == 'validation':
            list_val.append(data['midi_filename'][k])
        elif data['split'][k] == 'test':
            list_test.append(data['midi_filename'][k])

    if os.getcwd() not in save_dir:
        save_dir = os.path.join(os.getcwd(),save_dir)

    print(f'TOTAL TRAIN FILE: {len(list_train)}')
    print(f'TOTAL VAL FILE: {len(list_val)}')
    print(f'TOTAL TEST FILE: {len(list_test)}')
    print(f'PROCESSING...')
    
    # TRAIN
    train_data = multi_processing_data(list_train,data_dir,num_workers=args.num_workers,lth=args.lth,training=True)
    val_data  = multi_processing_data(list_val,data_dir,num_workers=1,lth=512,training=False)
    test_data = multi_processing_data(list_test,data_dir,num_workers=1,lth=512,training=False)

    print(f"TRAIN DATA SHAPE: {train_data.shape}")
    print(f"VAL DATA SHAPE: {val_data.shape}")
    print(f"TEST DATA SHAPE: {test_data.shape}")
    torch.save(train_data,os.path.join(save_dir,f"train_data.pt"))
    torch.save(val_data,os.path.join(save_dir,f"val_data.pt"))
    torch.save(test_data,os.path.join(save_dir,f"test_data.pt"))
    print(f"SAVE AUG DATA TO {os.path.join(save_dir,f'train_data.pt')}")