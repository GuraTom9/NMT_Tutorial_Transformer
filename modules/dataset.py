import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data
    
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        enc_in  = self.src_data[idx]
        dec_in  = [1] + self.tgt_data[idx]
        dec_out = self.tgt_data[idx] + [2]
        return enc_in, dec_in, dec_out

# Padding処理
def paired_collate_fn(insts):
    enc_in, dec_in, dec_out = list(zip(*insts))
    enc_in  = collate_fn(enc_in)
    dec_in  = collate_fn(dec_in)
    dec_out = collate_fn(dec_out)
    return (enc_in, dec_in, dec_out)

def collate_fn(insts):
    max_len = max(len(inst) for inst in insts)
    
    seq = []
    for inst in insts:
        element = inst + [0] * (max_len-len(inst))
        seq.append(element)
    
    seq = torch.LongTensor(seq)
    
    return seq