# -*- coding:utf-8 -*-
# Train Word Embbeding use Gensim

import argparse
import logging
import os
import sys

import numpy as np
from gensim.models import KeyedVectors
from gensim.models import word2vec

sys.path.append('../')
here = os.path.abspath(os.path.dirname(__file__))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__file__)


def train_wv(args):
    if not os.path.exists(args.file_path):
        sys.exit(-1)
    sentence = word2vec.LineSentence(args.file_path)
    model = word2vec.Word2Vec(sentence, sg=0, size=300, window=5, sample=0.0001,
                              min_count=3, workers=8, iter=20, compute_loss=True)

    model.wv.save_word2vec_format(args.save_path)


def use_wv(args):
    model = KeyedVectors.load_word2vec_format(args.file_path)
    for key in model.wv.similar_by_word(u'习近平', topn=10):
        print(key[0], key[1])


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

    print('{} {}'.format(vocab.size(), embed_size))
    for token, id in vocab.token2id.items():
        if token in model:
            print('{} {}'.format(token, ' '.join(map(str, model[token]))))
        else:
            emb = np.random.random((embed_size,)) - 0.5
            print('{} {}'.format(token, ' '.join(map(str, emb))))


def extract(args):
    from tokenizer import load_vocab
    logger.info('load vocab from {}'.format(args.vocab_path))

    vocab = load_vocab(vocab_path=args.vocab_path)
    logger.info('vocab size: {}'.format(vocab.size()))
    load_pretrain_embedding(vocab, embed_size=args.embed_size, embedding_path=args.file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='/home/demo1/womin/datasets/twlt/lt.data')
    parser.add_argument('--save_path', type=str, default='lt.wv.txt')
    parser.add_argument('--vocab_path', type=str, default='../data/vocab/vocab.txt')
    parser.add_argument('--embed_size', type=int, default=300)
    args = parser.parse_args()
    # train_wv(args)
    extract(args)
