# -*- coding:utf-8 -*-
# Date: 2019/7/18 14:26
# Author: xuxiaoping
# Desc: Dataset Loader

import codecs
import os

import numpy as np
from torch.utils.data import Dataset

from .tokenization import FullTokenizer
from .tokenizer import text_to_ids
import pandas as pd
import torch as t
from torchtext import data



class SnDataset(Dataset):
    """
    Sentiment dataset
    """

    def __init__(self, data_path, vocab=None, opt=None):
        self.maxlen = opt.maxlen
        self.vocab = vocab
        self.opt = opt
        self.data = self._load_dataset(data_path)

    def _load_dataset(self, data_path):
        assert os.path.exists(data_path)

        all_data = []
        with codecs.open(data_path, 'r', encoding='utf-8') as fin:
            for lidx, line in enumerate(fin):
                source, target = line[:-1].split('\t')
                source = "sos {0} eos".format(source)
                target = "sos {0} eos".format(target)

                source_raw_indices, source_length = text_to_ids(vocab=self.vocab,
                                                       tokens= source.strip().split(" "),
                                                       maxlen=self.opt.maxlen)

                target_raw_indices, target_length = text_to_ids(vocab=self.vocab,
                                                                tokens=target.strip().split(" "),
                                                                maxlen=self.opt.maxlen)


                data = {
                    'source_raw_indices': source_raw_indices,
                    'source_length': source_length,
                    'target_raw_indices': target_raw_indices,
                    'target_length': target_length
                }

                all_data.append(data)

        print('Load data from {}, data len:{}'.format(data_path, len(all_data)))

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

from data.text_utils import tokenizer

class MyDataset(data.Dataset):
    def __init__(self, path,text_field,len_field, test=False, aug=False, **kwargs):


        fields = [("id", None),
                  ("source_text", text_field),
                  ("source_length", len_field),
                  ("target_text", text_field),
                  ("target_len", len_field), ]
        examples = []


        with codecs.open(path) as fin:
            for index, line in enumerate(fin):


                if len(line[:-1].split("\t")) > 2:
                    source = " ".join(line[:-1].split("\t")[:-1])
                    target = line[:-1].split("\t")[-1]
                else:
                    source, target = line[:-1].split("\t")

                source_len = len(tokenizer(source)) + 2
                target_len = len(tokenizer(target)) + 2



                if target_len > text_field.fix_length or source_len > text_field.fix_length:
                    continue


                examples.append(data.Example.fromlist([None, source, source_len, target, target_len], fields))


        super(MyDataset, self).__init__(examples, fields, **kwargs)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)

