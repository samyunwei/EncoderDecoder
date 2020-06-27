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

class MyDataset(data.Dataset):
    def __init__(self, text_field, label_field, test=False, aug=False, **kwargs):

        fields = [("id", None),
                  ("source_text", text_field),
                  ("source_length", label_field)
                  ]
        examples = []

        source = "happy!"

        source_len = len(tokenizer(source))

        examples.append(data.Example.fromlist([None, source, source_len], fields))
        super(MyDataset, self).__init__(examples, fields, **kwargs)





if __name__ == '__main__':
    args = argument_parser()
    with open("seq2seq/TEXT.Field", "rb")as f:
        TEXT = dill.load(f)

    LENGTH = data.Field(sequential=False, use_vocab=False)

    embeddings = np.random.random((len(TEXT.vocab.itos), args.embed_size))
    args.TEXT = TEXT

    encoder = SN_MODELS["encoder"](embeddings, args)
    # atten = SN_MODELS["attention"](args.hidden_size * 4, 300)
    #decoder = SN_MODELS["decoder"](embeddings, args)
    atten = SN_MODELS["attention"](args.hidden_size, 2, "general")
    decoder = SN_MODELS["decoder"](embeddings, args, atten)


    model_class = SN_MODELS[args.model_name]

    model = model_class(encoder, decoder)

    checkpoint = t.load(args.load_dir)
    model.load_state_dict(checkpoint['model'])

    device = t.device('cuda' if False else 'cpu')

    greedy_model = SN_MODELS["greedy"](model.encoder, model.decoder, device, args)
    greedy_model.eval()


    val_iter = data.Iterator(MyDataset(TEXT, LENGTH), batch_size=1, sort_key=lambda x: len(x.Phrase),
                                   shuffle=True, device=-1)


    end_num = TEXT.vocab.stoi["<eos>"]

    for batch_id, batch_data in enumerate(val_iter, 1):
        result,_ = greedy_model(batch_data.source_text, batch_data.source_length)

        result = result.detach().numpy()
        result = result[ : result.tolist().index(end_num) ]

        for ele in result:
           print( TEXT.vocab.itos[ele] )

