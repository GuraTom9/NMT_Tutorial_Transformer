clean_corpus=/home/ogura/workspace/tutorial_nmt/mosesdecoder/scripts/training/clean-corpus-n.perl

perl -C ${clean_corpus} data/train-1 ja.txt en.txt data/train-1_cut150 1 150