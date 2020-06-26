# -*- coding:utf-8 -*-
# Date: 2019/7/18 14:26
# Author: xuxiaoping
# Desc: Tokenizer and Vocab

import codecs
import os
import re

import numpy as np
from gensim.models import KeyedVectors


def load_mood_dict():
    mood_dict = {}
    dict_tables = ['angry', 'disgusted', 'happy', 'sad', 'scared']
    for col in dict_tables:
        data_path = os.path.join('data/dictionary', '{}.txt'.format(col))
        with codecs.open(data_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                mood_dict[line.strip()] = 1
    print("mood dict len", len(mood_dict))


def text_to_ids(vocab, tokens, maxlen):
    def pad_input(sentence, seq_len):
        length = len(sentence)
        feature = np.zeros(seq_len, dtype=int)
        if len(sentence) != 0:
            feature[:length] = sentence[:seq_len]
        return feature,length

    unk = vocab.get_id(vocab.unk_token)
    setence = [vocab.token2id[word] if word in vocab.token2id else unk for word in tokens]

    sentences,length = pad_input(setence, maxlen)
    return sentences,length

    """Convert text to ids

    Args:
        vocab (Vocab):  class of Vocab
        tokens(list):
        maxlen(int): max ids length

    Returns:
        sequence: id list
        length: sequence length (before padding)

    unk = vocab.get_id(vocab.unk_token)
    pad = vocab.get_id(vocab.pad_token)
    global MOOD_DICT

    sequence = []
    for i, token in enumerate(tokens, 0):
        if i >= maxlen:
            break
        if token in vocab.token2id:
            token_id = vocab.token2id[token]
        else:
            token_id = unk
        sequence.append(token_id)

    length = len(sequence)
    # Padding
    if length < maxlen:
        sequence = sequence + list([pad] * (maxlen - len(sequence)))
    else:
        sequence = sequence[:maxlen]

    return np.array(sequence, dtype='int64'), length
    """

def recover_from_ids(vocab, ids):
    """
    Convert a list of ids to tokens
    Args:
        vocab (Vocab):  class of Vocab
        ids (list): a list of ids to convert
    Returns:
        a list of tokens
    """
    tokens = []
    for token_id in ids:
        tokens.append(vocab.get_token(token_id))
    return tokens


def load_pretrain_embedding(vocab, embed_size, embedding_path):
    """Load pretrain embedding from embedding_path

    Args:
        vocab (Vocab): class of Vocab
        embed_size: embedding size
        embedding_path: pretrain embedding path

    Returns:
        embeddings
    """
    model = KeyedVectors.load_word2vec_format(embedding_path)

    embeddings = np.random.random((vocab.size(), embed_size))
    for token, id in vocab.token2id.items():
        try:
            embeddings[id] = model[token]
        except KeyError as e:
            continue

    return embeddings


class Vocab:
    """Store vocab"""

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.token2cnt = {}
        self.pad_token = 'PAD'
        self.unk_token = 'UNK'

        #主要是为了text_to_ids方法中的token->id 填充与未知单词
        self.initial_tokens = [self.pad_token, self.unk_token]
        for token in self.initial_tokens:
            self.add(token)

    def add(self, token, cnt=1):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

        else:
            idx = self.token2id[token]

        if cnt > 0:
            if token in self.token2cnt:
                self.token2cnt[token] += cnt
            else:
                self.token2cnt[token] = cnt

        return idx

    def get_id(self, token):
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def get_token(self, idx):
        """
        gets the token corresponding to idx, returns unk token if idx is not in vocab
        Args:
            idx: an integer
        returns:
            a token string
        """
        try:
            return self.id2token[idx]
        except KeyError:
            return self.unk_token

    def filter_token_by_cnt(self, min_cnt):
        """
        filter the tokens in vocab by their count
        Args:
            min_cnt: tokens with frequency less than min_cnt is filtered
        """
        filtered_tokens = [
            token for token in self.token2id if self.token2cnt[token] >= min_cnt
        ]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}

        for token in self.initial_tokens:
            self.add(token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)

    def size(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)

    def show(self):
        dict_slices = lambda data, start, end: {k: data[k] for k in list(data.keys())[start:end]}

        print(dict_slices(self.token2id, 0, 50))



def load_vocab(vocab_path):
    vocab = Vocab()
    with open(vocab_path, 'r', encoding='utf-8') as fin:
        for lidx, line in enumerate(fin):
            try:
                token, id = line.strip().split('\t')
            except ValueError:
                print(token),id
            vocab.token2id[token] = int(id)
            vocab.id2token[int(id)] = token

    return vocab


def save_vocab(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as fout:
        for token, idx in vocab.token2id.items():
            fout.write('{}\t{}\n'.format(token, idx))


def load_stop_words():
    words = {}
    print("load stop words")
    with codecs.open('data/stop_words.dict', 'r', encoding='utf-8') as fin:
        for line in fin:
            word = line.strip()
            words[word] = 1

    return words


def create_vocab(args):
    """Create Vocab"""
    vocab = Vocab()
    LINE_PATTERN = r'[^\u4e00-\u9fa5]'
    stop_words = load_stop_words()

    with codecs.open(args.file_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            cols = line.strip().split('\t')

            for token in cols[1].split():
                token = re.sub(LINE_PATTERN, '', token)
                if len(token) > 4 or token in stop_words:
                    continue

                if token.strip():
                    vocab.add(token)

    print("vocab len: {}".format(vocab.size()))
    vocab.filter_token_by_cnt(min_cnt=args.min_cnt)
    print("after filter vocab len: {}".format(vocab.size()))
    save_vocab(vocab, args.save_path)
    # vocab.save(args.save_path)
    print("Number token:{}".format(vocab.size()))

    return vocab


def test_create_vocab(args):


    create_vocab(args)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--file_path', default='data/20190612/6class_token_data.txt', type=str, help='embedding path')
    parser.add_argument('--save_path', default='data/vocab/vocab.txt', type=str, help='save path')
    parser.add_argument('--min_cnt', default=2, type=int, help='min count')

    #test load_vocab
    #parser.add_argument('--min_cnt', default=2, type=int, help='min count')
    parser.add_argument('--vocab_path',
        default = '/home/demo1/womin/piguanghua/data/Amazon_Reviews_for_Sentiment_Analysis/vocab.txt',
        type = str, help = 'vocab_path')

    args = parser.parse_args()
    vocab = load_vocab(args.vocab_path)
    vocab.show()



