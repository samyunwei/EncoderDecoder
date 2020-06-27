# -*- coding:utf-8 -*-
# Date: 2019/7/18 14:26
# Author: xuxiaoping
# Desc: Model Zoo


from .no_attention_encoder_decoder import PlainSeq2Seq,PlainDecoder,PlainEncoder
from .attention_encoder_decoder import Encoder,AttenDecoder,AttenSeq2Seq,Attention
from .criterion import  LanguageModeCriterion
from .greedy_encoder_docoder import GreedySearchDecoder


SN_MODELS = {
    'lm': AttenSeq2Seq,
    'encoder': Encoder,
    'decoder': AttenDecoder,
    'criterion': LanguageModeCriterion,
    "greedy": GreedySearchDecoder,
    'attention': Attention
}


'''
SN_MODELS = {
    'lm': PlainSeq2Seq,
    'encoder': PlainEncoder,
    'decoder': PlainDecoder,
    'criterion': LanguageModeCriterion
}
'''
