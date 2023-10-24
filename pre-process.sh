TOKENIZER=/home/ogura/workspace/tutorial_nmt/mosesdecoder/scripts/tokenizer/tokenizer.perl
KYTEA_MODEL=/home/ogura/workspace/tutorial_nmt/kytea/models/jp-0.4.7-1.mod
Z2H=/home/ogura/workspace/tutorial_nmt/script.converter.distribution/z2h-utf8.pl


# echo "[Info] Tokenize English data..."
# for file in train-1 dev devtest test; do
#     cat /home/ogura/workspace/NMT_Tutorial_Transformer/pre_data/${file}.en.txt | \
#     perl -C $Z2H | \
#     perl -C $TOKENIZER -threads 4 -l en -no-escape > /home/ogura/workspace/NMT_Tutorial_Transformer/tok_data/${file}.en.txt
# done

echo "[Info] Tokenize Japanese data..."
for file in train-1 dev devtest test; do
    cat /home/ogura/workspace/NMT_Tutorial_Transformer/pre_data/${file}.ja.txt | \
    perl -C -pe 'use utf8; s/　/ /g;' | \
    kytea -model $KYTEA_MODEL -out tok | \
    perl -C -pe 's/^ +//; s/ +$//; s/ +/ /g;' | \
    perl -C -pe 'use utf8; tr/\|[]/｜［］/; ' > /home/ogura/workspace/NMT_Tutorial_Transformer/tok_data/${file}.ja.txt
done
