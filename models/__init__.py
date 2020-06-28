# -*- coding:utf-8 -*-
# Date: 2019/7/18 14:26
# Author: xuxiaoping
# Desc: Model Zoo


from .no_attention_encoder_decoder import PlainSeq2Seq,PlainDecoder,PlainEncoder
from .attention_encoder_decoder import Encoder,AttenDecoder,AttenSeq2Seq,Attention
from .criterion import  LanguageModeCriterion
from .greedy_encoder_docoder import GreedySearchDecoder
from .test_encoder_decoder import EncoderRNN,Attn,LuongAttnDecoderRNN,TestEncoderDecoder



SN_MODELS = {
    'lm': TestEncoderDecoder,
    'encoder': EncoderRNN,
    'decoder': LuongAttnDecoderRNN,
    'criterion': LanguageModeCriterion,
    "greedy": GreedySearchDecoder,
    'attention': Attn
}


'''
SN_MODELS = {
    'lm': PlainSeq2Seq,
    'encoder': PlainEncoder,
    'decoder': PlainDecoder,
    'criterion': LanguageModeCriterion
}
'''
