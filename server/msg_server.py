# -*- coding: utf-8 -*-
# @Time    : 2020/6/28 15:41
# @Author  : piguanghua
# @FileName: server.py
# @Software: PyCharm


from data import text_utils
import numpy as np
import codecs
import numpy as np
import dill

with open("seq2seq/TEXT.Field", "rb") as f:
    TEXT = dill.load(f)

text_cos = np.load("seq2seq/text_cos.npy")

word_vector_path = "train_data/craw1.npz.npz"
word_vector = np.load(word_vector_path, allow_pickle=True)["embeddings"]

target_file = "train_data/bakeup_chat_target.txt"
target_text = []
with codecs.open(target_file, "r") as fin:
    for line in iter(fin):
        target_text.append(" ".join(text_utils.tokenizer(line)))

def return_msg(msg):
    tokens = text_utils.tokenizer(msg)

    msg_cos = 0
    for token in tokens:
        msg_cos += word_vector.item().weight[TEXT.vocab.stoi[token]]

    scores = np.dot(msg_cos, text_cos.T)
    index = np.argmax(scores)
    return target_text[index]