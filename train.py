import datetime
import argparse
import json
from logging import getLogger, INFO, FileHandler, Formatter
import os
from tqdm import tqdm
import nltk
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import modules.dataset as dataset
import modules.models as models
from modules.PositionalEncoding import PositionalEncoding
from modules.translate import translate
from modules.MyNLLLoss import MyNLLLoss
from modules.preprocess import create_dictionary, token2idx, create_idx_pair, load_data_from_pickle


def train(PAD, BOS, EOS, max_len, dictionary, pos_enc, epoch_size, d_model,
          train_loader, dev_loader, dev_word_data, transformer,
          criterion, optimizer, scaler, device, model_name):

    #train_logger = getLogger(__name__).getChild("train")

    max_score = 0
    step_num = 1
    warmup_steps = 4000
    max_norm = 5.0
    use_amp = "store_true"

    logsoftmax = nn.LogSoftmax(dim=2)

    for epoch in range(epoch_size):
        
        transformer.train()

        pbar = tqdm(train_loader, ascii=True)
        total_loss = 0

        for i, batch in enumerate(pbar):
            
            optimizer.zero_grad()

            enc_in, dec_in, dec_out = map(lambda x: x.to(device), batch)
            
            batch_size = enc_in.size(0)
            src_sent_len, tgt_sent_len = enc_in.size(1), dec_in.size(1)
            src_pad_mask = enc_in.eq(PAD).unsqueeze(1) 
            tgt_pad_mask = dec_in.eq(PAD).unsqueeze(1)

            enc_self_attn_mask = src_pad_mask.expand(-1, src_sent_len, -1)
            dec_self_attn_mask = tgt_pad_mask.expand(-1, tgt_sent_len, -1)
            infer_mask = torch.ones((tgt_sent_len, tgt_sent_len),
                                    dtype=torch.uint8,
                                    device=device).triu(diagonal=1)
            infer_mask = infer_mask.unsqueeze(0).expand(batch_size, -1, -1)
            dec_self_attn_mask = torch.gt(dec_self_attn_mask + infer_mask, 0)
            dec_src_tgt_mask = src_pad_mask.expand(-1, tgt_sent_len, -1)
            
            enc_pos_enc = pos_enc[:src_sent_len, :].unsqueeze(0)
            enc_pos_enc = enc_pos_enc.expand(batch_size, -1, -1).to(device)
            dec_pos_enc = pos_enc[:tgt_sent_len, :].unsqueeze(0)
            dec_pos_enc = dec_pos_enc.expand(batch_size, -1, -1).to(device)
            
            with autocast():
                
                output = transformer(enc_in, dec_in, enc_pos_enc, dec_pos_enc,
                                     enc_self_attn_mask, dec_self_attn_mask,
                                     dec_src_tgt_mask)
                
                output = logsoftmax(output)
                output = output.view(-1, output.size(-1))
                dec_out = dec_out.view(-1)
                
                loss = criterion(output, dec_out)

            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=max_norm)

            lrate = d_model**(-0.5) * min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
            for op in optimizer.param_groups:
                op["lr"] = lrate
            scaler.step(optimizer)
            scaler.update()
            step_num += 1

            pbar.set_description("[epoch:%d] loss:%f" % (epoch+1, total_loss/(i+1)))

        if epoch >= 0:
            sentences = translate(PAD, BOS, EOS,
                                  max_len, dictionary, pos_enc,
                                  dev_loader, transformer, device)
            bleu_score = nltk.translate.bleu_score.corpus_bleu(dev_word_data, sentences) * 100
            print("BLEU:", bleu_score)
            if bleu_score > max_score:
                max_score = bleu_score
                torch.save(transformer.state_dict(), model_name)
                print("saved best model.")
            if epoch_size - epoch <= 5:
                torch.save(transformer.state_dict(), "./model/check_point/check_point_{}.pt".format(5 - (epoch_size - epoch)))
                print("saved check point model{}.".format(5 - (epoch_size - epoch)))
            
        # train_logger.info("[epoch:%d] loss:%f BLEU:%f"
        # % (epoch+1, total_loss/(i+1), bleu_score))


def main():
    SRC_LANG = "en"
    TGT_LANG = "ja"

    TRAIN_SRC_CORPUS_PATH = "data/train-1_top100000.{}.txt".format(SRC_LANG)
    TRAIN_TGT_CORPUS_PATH = "data/train-1_top100000.{}.txt".format(TGT_LANG)

    DEV_SRC_CORPUS_PATH = "data/dev.{}.txt".format(SRC_LANG)
    DEV_TGT_CORPUS_PATH = "data/dev.{}.txt".format(TGT_LANG)

    TEST_SRC_CORPUS_PATH = "data/test.{}.txt".format(SRC_LANG)
    TEST_TGT_CORPUS_PATH = "data/test.{}.txt".format(TGT_LANG)

    # ハイパーパラメータの設定
    max_length = 150
    sentence_num = 100000
    batch_size = 128
    epoch_size = 20
    learning_rate = 0.001
    dropout = 0.1 
    d_ff = 2048
    d_model = 512
    init = True
    label_smoothing = 0.1
    #max_norm = 5.0
    name = "check_point"
    parallel_size = 8
    seed = 42
    sub_layer_num = 6
    #use_amp = "store_true"
    valid_batch_size = 50
    weight_decay = 1e-5

    random.seed(seed)
    torch.manual_seed(seed)

    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    # save config file
    save_dir = "./model/check_point"
    # save_dir = "./model/{}_{}".format(name, datetime_str) if name != "no_name" else "./model/no_name"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # with open("{}/config.json".format(save_dir, name), mode="w") as f:
    #     json.dump(vars(args), f, separators=(",", ":"), indent=4)

    # logger = getLogger(__name__)
    # log_name = "{}/train.log".format(save_dir)
    model_name = "{}/best.pt".format(save_dir)
        
    # fh = FileHandler(log_name)
    # fmt = Formatter("[%(levelname)s] %(asctime)s (%(name)s) - %(message)s")
    # logger.setLevel(INFO)
    # fh.setLevel(INFO)
    # fh.setFormatter(fmt)
    # logger.addHandler(fh)
    
    # logger.info(args)
    # print(args)

    #####

    # 原言語文データの読み込み
    src_train_file = TRAIN_SRC_CORPUS_PATH
    src_dev_file =  DEV_SRC_CORPUS_PATH
    src_test_file = TEST_SRC_CORPUS_PATH

    src2idx = create_dictionary(src_train_file)
    with open("data/src2idx.pkl", 'wb') as f:
        pickle.dump(src2idx, f)

    src_train_idx = token2idx(src_train_file, src2idx)
    src_dev_idx = token2idx(src_dev_file, src2idx)
    src_test_idx = token2idx(src_test_file, src2idx)

    # 目的言語文データの読み込み
    tgt_train_file = TRAIN_TGT_CORPUS_PATH
    tgt_dev_file = DEV_TGT_CORPUS_PATH
    tgt_test_file = TEST_TGT_CORPUS_PATH

    tgt2idx = create_dictionary(tgt_train_file)
    with open("data/tgt2idx.pkl", 'wb') as f:
        pickle.dump(tgt2idx, f)

    tgt_train_idx = token2idx(tgt_train_file, tgt2idx)
    tgt_dev_idx = token2idx(tgt_dev_file, tgt2idx)
    tgt_test_idx = token2idx(tgt_test_file, tgt2idx)

    idx2src = {v: k for k, v in src2idx.items()}
    idx2tgt = {v: k for k, v in tgt2idx.items()}
    BOS = src2idx["<BOS>"]
    EOS = src2idx["<EOS>"]
    PAD = src2idx["<PAD>"]
    src_dict_size = len(src2idx)
    tgt_dict_size = len(tgt2idx)

    create_idx_pair(src_train_idx, tgt_train_idx, "data/ja-en_train_idx.pkl")
    create_idx_pair(src_dev_idx, tgt_dev_idx, "data/ja-en_dev_idx.pkl")
    create_idx_pair(src_test_idx, tgt_test_idx, "data/ja-en_test_idx.pkl")

    src_train_idx_list, tgt_train_idx_list = load_data_from_pickle("data/ja-en_train_idx.pkl")
    src_dev_idx_list, tgt_dev_idx_list = load_data_from_pickle("data/ja-en_dev_idx.pkl")
    src_test_idx_list, tgt_test_idx_list = load_data_from_pickle("data/ja-en_test_idx.pkl")

    exit()

    # データローダーの作成
    train_data = dataset.MyDataset(src_data=src_train_idx_list, tgt_data=tgt_train_idx_list)
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=dataset.paired_collate_fn, shuffle=True)

    dev_data = dataset.MyDataset(src_data=src_dev_idx_list, tgt_data=tgt_dev_idx_list)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, collate_fn=dataset.paired_collate_fn, shuffle=False)

    with open(DEV_TGT_CORPUS_PATH, "r") as f:
        lines = f.readlines()
    tgt_dev_word_list = [ line.strip().split(" ") for line in lines ] 
    dev_word_data = [ [words] for words in tgt_dev_word_list ]

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ######
    
    pos_enc = PositionalEncoding(max_length+100, d_model)

    transformer = models.Transformer(
        PAD,
        d_model, d_ff,
        src_dict_size, tgt_dict_size,
        parallel_size, sub_layer_num,
        dropout,
        init
    ).to(device)

    optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay)
    
    scaler = GradScaler()

    criterion = MyNLLLoss(smooth_weight=label_smoothing, ignore_index=PAD)

    train(PAD, BOS, EOS, max_length, idx2tgt, pos_enc, epoch_size, d_model,
          train_loader, dev_loader, dev_word_data, transformer,
          criterion, optimizer, scaler, device, model_name)


if __name__ == "__main__":
    main()