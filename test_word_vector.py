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
import torch as t

if __name__ == '__main__':
    pretrain = "./train_data/word_vector.npz"
    embeddings = torch.tensor(
        np.load(pretrain, allow_pickle=True)["embeddings"].item().weight)

    embeddings