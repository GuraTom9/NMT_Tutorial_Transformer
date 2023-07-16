import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, head_num, dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.head_num = head_num
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K_V, mask):

        d_model = self.d_model
        head_num = self.head_num
        
        Q = self.linear_Q(Q)    # [batch_size, seq_len, d_model]
        K = self.linear_K(K_V)
        V = self.linear_V(K_V)

        Q = torch.chunk(Q, self.head_num, dim=2)    #[batch_size, seq_len, d_model/head_num]
        K = torch.chunk(K, self.head_num, dim=2)
        V = torch.chunk(V, self.head_num, dim=2)

        Q = torch.cat(Q, dim=0) # [batch_size * head_num, seq_len, d_model/head_num]
        K = torch.cat(K, dim=0)
        V = torch.cat(V, dim=0)
        
        QK = torch.bmm(Q, torch.transpose(K, 1, 2)) 
        QK = torch.div(QK, (d_model/head_num)**0.5) # [batch_size * head_num, seq_len, seq_len]
        
        masks = mask.repeat(head_num, 1, 1)
        QK.masked_fill_(masks, -float("inf"))
        
        softmax_QK = self.softmax(QK)
        softmax_QK = self.dropout(softmax_QK)

        QKV = torch.bmm(softmax_QK, V)  # [batch_size * head_num, seq_len, d_model/head_num]
        QKV = torch.chunk(QKV, head_num, dim=0)
        QKV = torch.cat(QKV, dim=2) # [batch_size, seq_len, d_model]
        QKV = self.linear(QKV)
        
        return QKV