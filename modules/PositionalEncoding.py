import torch


def PositionalEncoding(seq_len, d_model):
    pos_num = torch.arange(seq_len, dtype=torch.float)
    pos_num = pos_num.unsqueeze(dim=1).expand(-1, d_model)  # [seq_len, d_model]
    
    dim_num = torch.arange(d_model,dtype=torch.float)
    dim_num = dim_num.unsqueeze(dim=0).expand(seq_len, -1)  # [seq_len, d_model]
    
    bottom = torch.where(dim_num%2==0, torch.pow(10000, dim_num/d_model), torch.pow(10000, (dim_num-1)/d_model))
    pe = torch.where(dim_num%2==0, torch.sin(pos_num/bottom), torch.cos(pos_num/bottom))
    
    return pe
 