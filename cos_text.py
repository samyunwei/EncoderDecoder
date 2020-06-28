# -*- coding: utf-8 -*-
# @Time    : 2020/6/28 15:43
# @Author  : piguanghua
# @FileName: cos_text.py
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
from data import text_utils

if __name__ == '__main__':
    with open("seq2seq/TEXT.Field", "rb") as f:
        TEXT = dill.load(f)

    source_path = "train_data/bakeup_chat_source.txt"
    word_vector_path = "train_data/word_vector.npz"

    text_cos = []
    word_vector = np.load(word_vector_path, allow_pickle=True)["embeddings"]

    with codecs.open(source_path, "r") as sfin:
        for index,line in enumerate(sfin):
            tokens = text_utils.tokenizer(line)


            token_cos = 0
            for token in tokens:
               token_cos += word_vector.item().weight[TEXT.vocab.stoi[token]]
            text_cos.append(token_cos)

    import numpy as np

    for index,ele in enumerate(text_cos) :
        text_cos[index] = ele.numpy()


    np.save("seq2seq/text_cos.npy", text_cos)




