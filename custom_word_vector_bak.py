# -*- coding: utf-8 -*-
# @Time    : 2020/6/28 10:01
# @Author  : piguanghua
# @FileName: custom_word_vector.py
# @Software: PyCharm
import codecs
import numpy as np
import dill
from torchtext import data
import re
import jieba
from collections import Counter
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import torch
import torch.nn as nn
import pickle as pkl
from gensim.test.utils import common_texts
from gensim.models import Word2Vec


if __name__ == '__main__':
    with open("seq2seq/TEXT.Field", "rb")as f:
        TEXT = dill.load(f)
    TEXT
    glove_path = "/home/demo1/crawl-300d-2M.vec"
    weight = torch.zeros(len(TEXT.vocab), 300)

    word2idx = {o: i for i, o in enumerate(TEXT.vocab.itos)}
    idx2word = {i: o for i, o in enumerate(TEXT.vocab.itos)}

    word_to_idx = word2idx
    idx_to_word = idx2word


    i = 0
    with codecs.open(glove_path) as fin:
        for index, line in enumerate(fin):
            if index == 0: continue

            word_index = len(line.split(" ")) - 300


            data = line.split(" ")[word_index:]
            vector = np.array(list(map(lambda x: float(x), data)))

            word = line.split(" ")[:word_index][0]
            try:
                index = word_to_idx[word]
                print("line = {0}".format(index))
            except KeyError as e:
                continue
            else:
                weight[index, :] = torch.from_numpy(vector)
                i+=1
                print(i)

                if word == "<unk>":
                    print(torch.from_numpy(vector))
                if word == "<pad>":
                    print(torch.from_numpy(vector))
                if word == "<sos>":
                    print(torch.from_numpy(vector))

                if i == len(TEXT.vocab) -1 :
                    break


    embedding = nn.Embedding.from_pretrained(weight)
    filename_trimmed_dir = "/home/demo1/womin/piguanghua/pycharm/EncoderDecoder/seq2seq/craw1.npz"
    np.savez_compressed(filename_trimmed_dir, embeddings=embedding)