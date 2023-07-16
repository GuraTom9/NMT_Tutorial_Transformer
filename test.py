import nltk
import pickle

import torch
from torch.utils.data import DataLoader

import modules.dataset as dataset
import modules.models as models
from modules.PositionalEncoding import PositionalEncoding
from modules.translate import translate
from modules.BeamSearchTranslate import BeamSearchTranslate
from modules.preprocess import load_data_from_pickle


def main():
    TGT_LANG = "ja"
    
    TEST_TGT_CORPUS_PATH = f"data/test.{TGT_LANG}.txt"

    batch_size = 64
    dropout = 0.1
    d_ff = 2048
    d_model = 512
    init = True
    parallel_size = 8
    sub_layer_num = 6

    beam_search = True

    # モデル読み込み
    model_name = "model/check_point/avg_check_point.pt"

    name = "output"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    with open("data/src2idx.pkl", 'rb') as f:
        src2idx = pickle.load(f)
    with open("data/tgt2idx.pkl", 'rb') as f:
        tgt2idx = pickle.load(f)
    idx2src = {v: k for k, v in src2idx.items()}
    idx2tgt = {v: k for k, v in tgt2idx.items()}
    PAD = src2idx["<PAD>"]
    EOS = src2idx["<EOS>"]
    BOS = src2idx["<BOS>"]
    src_dict_size = len(src2idx)
    tgt_dict_size = len(tgt2idx)

    src_test_idx_list, tgt_test_idx_list = load_data_from_pickle("data/ja-en_test_idx.pkl")

    # データローダーの作成
    test_data = dataset.MyDataset(src_data=src_test_idx_list, tgt_data=tgt_test_idx_list)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=dataset.paired_collate_fn, shuffle=False)

    with open(TEST_TGT_CORPUS_PATH, "r") as f:
        lines = f.readlines()
    tgt_test_word_list = [ line.strip().split(" ") for line in lines ] 
    test_word_data = [ [words] for words in tgt_test_word_list ]

    src_max_len = max([len(element) for element in src_test_idx_list])
    pos_enc = PositionalEncoding(src_max_len+50, d_model)
    
    transformer = models.Transformer(
        PAD,
        d_model, d_ff,
        src_dict_size, tgt_dict_size,
        parallel_size, sub_layer_num,
        dropout,
        init
    ).to(device)
    
    transformer.load_state_dict(torch.load(model_name))
    
    if beam_search:
        sentence_list = BeamSearchTranslate(PAD, BOS, EOS,
                              src_max_len, idx2tgt, pos_enc,
                              test_loader, transformer, device)

    else:
        sentence_list = translate(PAD, BOS, EOS,
                              src_max_len, idx2tgt, pos_enc,
                              test_loader, transformer, device)
    
    bleu_score = nltk.translate.bleu_score.corpus_bleu(test_word_data, sentence_list) * 100
    print("BLEU:", bleu_score)
    
    sentences = ""
    for sentence in sentence_list:
        sentences += " ".join(sentence) + "\n"
    with open("{}.tok".format(name), mode="w") as output_f:
        output_f.write(sentences)


if __name__ == "__main__":
    main()