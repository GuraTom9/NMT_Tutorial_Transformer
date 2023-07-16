import pickle

# 訓練データから辞書作成
def create_dictionary(train_file):
    # 辞書の初期化
    token2idx= {"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3}
    index = 4

    with open(train_file, "r") as f:
        for line in f:
            line = line.strip().split()
            for token in line:
                if token not in token2idx:
                    token2idx[token] = index
                    index += 1
        return token2idx

# データのトークンをID化
def token2idx(text_file, token2idx):
    with open(text_file, "r") as f:
        lines = f.readlines()
    
    idx_lists = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        tokens = line.split()
        idx = [token2idx.get(token, token2idx["<UNK>"]) for token in tokens]
        idx_lists.append(idx)
    
    return idx_lists

# 2つのリストをペアにしてpickleファイルで保存
def create_idx_pair(list1, list2, save_path):
    pairs = []
    for idx_list1, idx_list2 in zip(list1, list2):
        pairs.append([idx_list1, idx_list2])
    with open(save_path, 'wb') as f:
        pickle.dump(pairs, f)

# pickleファイルからデータを読み込みリストに分割
def load_data_from_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return list(map(list, zip(*data)))
