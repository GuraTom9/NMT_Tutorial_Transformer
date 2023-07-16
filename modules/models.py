import torch
import torch.nn as nn

from modules.MultiHeadAttention import MultiHeadAttention
from modules.FeedForwardNetwork import FeedForwardNetwork


class Transformer(nn.Module):

    def __init__(self, pad_index, d_model, d_ff,
                 src_dict_size, tgt_dict_size, head_num,
                 sub_layer_num, dropout, init):
        super(Transformer, self).__init__()

        self.encoder = Encoder(pad_index, d_model, d_ff,
                               src_dict_size, head_num, sub_layer_num,
                               dropout, init)
        self.decoder = Decoder(pad_index, d_model, d_ff,
                               tgt_dict_size, head_num, sub_layer_num,
                               dropout, init)
        
    def forward(self, source, target, enc_pos_enc, dec_pos_enc,
                enc_self_mask, dec_self_mask, dec_src_tgt_mask):

        encoder_output = self.encoder(source, enc_pos_enc, enc_self_mask)
        decoder_output = self.decoder(target, encoder_output, dec_pos_enc,
                                         dec_self_mask, dec_src_tgt_mask)

        return decoder_output


class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_ff, head_num, dropout):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, head_num, dropout)
        self.FFN = FeedForwardNetwork(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, self_attn_mask):

        self_attn_out = self.self_attn(input, input, self_attn_mask)
        self_attn_out = self.dropout(self_attn_out)
        self_attn_out = self.layer_norm(input + self_attn_out)

        ffn_out = self.FFN(self_attn_out)
        ffn_out = self.dropout(ffn_out)
        output = self.layer_norm(self_attn_out + ffn_out)

        return output
    

class Encoder(nn.Module):

    def __init__(self, pad_index, d_model, d_ff, enc_dict_size,
                 head_num, sub_layer_num, dropout, init):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(enc_dict_size, d_model, padding_idx=pad_index)
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_sub_layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model, d_ff, head_num, dropout
                )
                for _ in range(sub_layer_num)
            ]
        )
        
        nn.init.constant_(self.embedding.weight[pad_index], 0)
        if init:
            nn.init.normal_(self.embedding.weight, mean=0, std=d_model ** -0.5)
        
    def forward(self, input, pos_enc, self_attn_mask):
        
        embedded = self.embedding(input)
        
        encoded = torch.mul(embedded, self.d_model**0.5) + pos_enc
        encoded = self.dropout(encoded)

        layer_input = encoded
        for encoder_sub_layer in self.encoder_sub_layers:
            layer_output = encoder_sub_layer(layer_input, self_attn_mask)
            layer_input = layer_output
        output = layer_output

        return output
    

class DecoderBlock(nn.Module):

    def __init__(self, d_model, d_ff, head_num, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, head_num, dropout)
        self.src_tgt_attn = MultiHeadAttention(d_model, head_num, dropout)
        self.FFN = FeedForwardNetwork(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, encoder_output, self_attn_mask, src_tgt_mask):
        
        self_attn_out = self.self_attn(input, input, self_attn_mask)
        self_attn_out = self.dropout(self_attn_out)
        self_attn_out = self.layer_norm(input + self_attn_out)

        src_tgt_attn_out = self.src_tgt_attn(self_attn_out, encoder_output, src_tgt_mask)
        src_tgt_attn_out = self.dropout(src_tgt_attn_out)
        src_tgt_attn_out = self.layer_norm(self_attn_out + src_tgt_attn_out)

        ffn_out = self.FFN(src_tgt_attn_out)
        ffn_out = self.dropout(ffn_out)
        output = self.layer_norm(src_tgt_attn_out + ffn_out)

        return output
    

class Decoder(nn.Module):

    def __init__(self, pad_index, d_model, d_ff, dec_dict_size,
                 head_num, sub_layer_num, dropout, init):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(dec_dict_size, d_model, padding_idx=pad_index)
        self.dropout = nn.Dropout(p=dropout)
        self.decoder_sub_layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model, d_ff, head_num, dropout
                )
                for _ in range(sub_layer_num)
            ]
        )
        self.out = nn.Linear(d_model, dec_dict_size, bias=False)
        # self.logsoftmax = nn.LogSoftmax(dim=2)
        
        nn.init.constant_(self.embedding.weight[pad_index], 0)
        if init:
            nn.init.normal_(self.embedding.weight, mean=0, std=d_model ** -0.5)
            nn.init.xavier_uniform_(self.out.weight)
              
    def forward(self, input, encoder_output, pos_enc, self_attn_mask, src_tgt_mask):

        embedded = self.embedding(input)

        encoded = torch.mul(embedded, self.d_model**0.5) + pos_enc
        encoded = self.dropout(encoded)
        
        layer_input = encoded
        for decoder_sub_layer in self.decoder_sub_layers:
            layer_output = decoder_sub_layer(layer_input, encoder_output, self_attn_mask, src_tgt_mask)
            layer_input = layer_output
        output = layer_output

        output = self.out(output)
        # output = self.logsoftmax(output)

        return output



# class MultiHeadAttention(nn.Module):

#     def __init__(self, d_model, head_num, dropout):
#         super(MultiHeadAttention, self).__init__()
#         self.d_model = d_model
#         self.head_num = head_num
#         self.query_linear = nn.Linear(d_model, d_model)
#         self.key_linear = nn.Linear(d_model, d_model)
#         self.value_linear = nn.Linear(d_model, d_model)
#         self.softmax = nn.Softmax(dim=2)
#         self.dropout = nn.Dropout(p=dropout)
#         self.output_linear = nn.Linear(d_model, d_model)
        
#     def forward(self, query, key_value, mask):

#         d_model = self.d_model
#         head_num = self.head_num
        
#         query = self.query_linear(query)
#         query = torch.chunk(query, self.head_num, dim=2)
#         query = torch.cat(query, dim=0)

#         key = self.key_linear(key_value)
#         key = torch.chunk(key, self.head_num, dim=2)
#         key = torch.cat(key, dim=0)

#         value = self.value_linear(key_value)
#         value = torch.chunk(value, self.head_num, dim=2)
#         value = torch.cat(value, dim=0)
        
#         key = torch.transpose(key, 1, 2)
#         score = torch.bmm(query, key)
#         score = torch.div(score, (d_model/head_num)**0.5)
#         masks = mask.repeat(head_num, 1, 1)
#         score.masked_fill_(masks, -float("inf"))
        
#         weight = self.softmax(score)
#         weight = self.dropout(weight)
#         heads = torch.bmm(weight, value)
#         heads = torch.chunk(heads, head_num, dim=0)
#         heads = torch.cat(heads, dim=2)
        
#         heads = self.output_linear(heads)
        
#         return heads


# class FeedForwardNetwork(nn.Module):

#     def __init__(self, input_size, d_model, output_size, dropout):
#         super(FeedForwardNetwork, self).__init__()
#         self.ffn1 = nn.Linear(input_size, d_model)
#         self.ffn2 = nn.Linear(d_model, output_size)
#         self.dropout = nn.Dropout(p=dropout)
        
#     def forward(self, input):
#         output = self.ffn1(input)
#         output = F.relu(output)
#         output = self.dropout(output)
#         output = self.ffn2(output)
#         return output
    