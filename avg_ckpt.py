import argparse
import copy
from collections import OrderedDict

import pickle

import torch
from torch import Tensor

import modules.models as models
from modules.PositionalEncoding import PositionalEncoding

def average_checkpoints(model, model_paths): # model_paths:chパス5つ "a:b:c:d:e"
    model_states = []
    for model_path in model_paths:
        model.load_state_dict(torch.load(model_path))
        model_state = copy.deepcopy(model.state_dict())
        model_states.append(model_state)

    params_sum = OrderedDict()
    for i, model_state in enumerate(model_states):
        if i == 0:
            for key, value in model_state.items():
                params_sum[key] = value.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
        else:
            for key, value in model_state.items():
                params_sum[key] += value

    averaged_state_dict = OrderedDict()
    for key, value in params_sum.items():
        averaged_state_dict[key] = value / len(model_states)

    return averaged_state_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths = "model/check_point/check_point_0.pt:model/check_point/check_point_1.pt:model/check_point/check_point_2.pt:model/check_point/check_point_3.pt:model/check_point/check_point_4.pt"
    model_paths = model_paths.split(":")

    output_path = "model/check_point/avg_check_point.pt"

    with open("/home/ogura/workspace/v2_transformer_nmt/data/src2idx.pkl", 'rb') as f:
        src2idx = pickle.load(f)
    with open("/home/ogura/workspace/v2_transformer_nmt/data/tgt2idx.pkl", 'rb') as f:
        tgt2idx = pickle.load(f)

    PAD = src2idx["<PAD>"]
    d_model = 512
    d_ff = 2048
    src_dict_size = len(src2idx)
    tgt_dict_size = len(tgt2idx)
    parallel_size = 8
    sub_layer_num = 6
    dropout = 0.1
    init = True

    transformer = models.Transformer(
        PAD,
        d_model, d_ff,
        src_dict_size, tgt_dict_size,
        parallel_size, sub_layer_num,
        dropout,
        init
    ).to(device)

    averaged_state_dict = average_checkpoints(transformer, model_paths)
    # model.load_state_dict(averaged_state_dict)

    torch.save(averaged_state_dict, output_path)
    print("saved averaged model.")

if __name__ == "__main__":
    main()