from args import argument_parser
from models import SN_MODELS
import torch as t
import torch.nn as nn
import dill

import spacy
spacy_en = spacy.load('en')

def tokenizer(text):  # create a tokenizer function
    # 返回 a list of <class 'spacy.tokens.token.Token'>
    return [tok.text for tok in spacy_en.tokenizer(text)]
from torchtext import data

import numpy as np
from data import text_utils

args = argument_parser()


with open("seq2seq/bak/TEXT.Field", "rb")as f:
    TEXT = dill.load(f)

LENGTH = data.Field(sequential=False, use_vocab=False)

embeddings = np.random.random((len(TEXT.vocab.itos), args.embed_size))
args.TEXT = TEXT

encoder = SN_MODELS["encoder"](embeddings, args)
# atten = SN_MODELS["attention"](args.hidden_size * 4, 300)
# decoder = SN_MODELS["decoder"](embeddings, args)
atten = SN_MODELS["attention"](args.hidden_size, "general")
decoder = SN_MODELS["decoder"](embeddings, args, atten)

model_class = SN_MODELS[args.model_name]

# model = model_class(encoder, decoder, args)
model = model_class(encoder, decoder, args)

checkpoint = t.load(args.load_dir)
model.load_state_dict(checkpoint['model'])

device = t.device('cuda' if False else 'cpu')

greedy_model = SN_MODELS["greedy"](model.encoder, model.decoder, device, args)
greedy_model.eval()

def model_msg(msg):
    tokens = ["<sos>"]
    tokens.extend(text_utils.tokenizer(msg))
    tokens.append("<eos>")

    tokens_ids = []
    for token in tokens:
        tokens_ids.append(TEXT.vocab.stoi[token])

    tensor_data = t.LongTensor(tokens_ids)
    source_text = t.unsqueeze(tensor_data, dim=0)

    length_data = t.LongTensor([len(tokens_ids)])
    source_length = length_data

    # source_text = source_text.to(device)
    # source_length = source_length.to(device)

    result, _ = greedy_model(source_text, source_length)

    random_key = np.random.randint(low=2, high=5, size=(1))[0]

    result = result[:random_key]

    return_msg = []
    for ele in result:
        if TEXT.vocab.itos[ele] != '<sos>' and TEXT.vocab.itos[ele] != '<eos>':
            return_msg.append(TEXT.vocab.itos[ele])

    return " ".join(return_msg)