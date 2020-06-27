
import spacy
spacy_en = spacy.load('en_core_web_sm')

def tokenizer(doc): # create a tokenizer function
    # 返回 a list of <class 'spacy.tokens.token.Token'>
    return [token.orth_ for token in spacy_en.tokenizer(doc) if not token.is_punct | token.is_space]
