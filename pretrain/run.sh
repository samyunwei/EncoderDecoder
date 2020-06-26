#!/usr/bin/env bash

# 清理语料
python raw_tools.py --clean_pretrain --file_path ./pretrain/kafka-corpus.txt > ./pretrain/clean_corpus.txt


# train word2vec
python gensim_word2vec.py

# train glove
./demo.sh

# 将glove格式转换为word2vec格式
python -m gensim.scripts.glove2word2vec --i -o glove.300d.txt

# train fastText
./fasttext cbow -input ../pretrain/clean_corpus.txt -minCount 5 -minn 1