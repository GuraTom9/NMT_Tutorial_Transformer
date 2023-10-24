#!/bin/bash

# 入力ファイルのパス
input_file="/home/ogura/workspace/NMT_Tutorial_Transformer/data/train-1_cut150.ja.txt"

# 出力ファイルのパス
output_file="/home/ogura/workspace/NMT_Tutorial_Transformer/data/train-1_top100000.ja.txt"

# 行数の上限
limit=100000

# 行数を数えるための変数
count=0

# 入力ファイルから一行ずつ読み込む
while read -r line; do
  # 行数が上限に達した場合はループを抜ける
  if [[ $count -ge $limit ]]; then
    break
  fi

  # 出力ファイルに行を書き込む
  echo "$line" >> "$output_file"

  # 行数をカウントする
  count=$((count+1))
done < "$input_file"
