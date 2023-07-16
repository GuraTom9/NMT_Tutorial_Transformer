import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
    