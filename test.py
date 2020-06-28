# -*- coding: utf-8 -*-
# @Time    : 2020/6/28 16:17
# @Author  : piguanghua
# @FileName: test.py
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
import numpy as np

if __name__ == '__main__':

    with open("seq2seq/TEXT.Field", "rb") as f:
        TEXT = dill.load(f)

    text_cos = np.load("seq2seq/text_cos.npy")

    word_vector_path = "seq2seq/craw1.npz"
    word_vector = np.load(word_vector_path, allow_pickle=True)["embeddings"]
    
    target_file = "train_data/bakeup_chat_target.txt"
    target_text = []
    with codecs.open(target_file, "r") as fin:
        for line in iter(fin):
            target_text.append( " ".join(text_utils.tokenizer(line)) )
        

    msg = "Is it cramped in the computer?"
    tokens = text_utils.tokenizer(msg)



    msg_cos = 0
    for token in tokens:
        msg_cos += word_vector.item().weight[TEXT.vocab.stoi[token.lower()]]

    scores = np.dot(msg_cos, text_cos.T)
    index = np.argmax(scores)
    print(target_text[index])


    
