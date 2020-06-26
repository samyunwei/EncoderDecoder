
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#paper:NEURAL MACHINE TRANSLATION_BY_JOINTLY LEARNING_TO_ALIGN_AND_TRANSLATE.pdf
#https://arxiv.org/pdf/1409.0473.pdf
class PlainEncoder(nn.Module):
    def __init__(self, embeddings, opts):
        super(PlainEncoder, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings=t.FloatTensor(embeddings),
                                                  freeze=False)
        self.net = nn.Sequential(
            nn.Dropout(opts.drop_rate),
            nn.GRU(opts.embed_size, opts.hidden_size, num_layers=opts.num_layers, batch_first=True,
                   bidirectional=True, dropout=opts.drop_rate)
        )

    '''
        text:batch,seq
        embed:batch,seq,embed
        gpu:batch,seq,embed
    '''
    def forward(self, text, length):
        '''
             (seq_len, batch, num_directions * hidden_size):
        '''
        embed = self.embed(text)
        output,h_n = self.net(embed)
        '''
           output: seq,batch,direction*hidden
           h_0: layer*direction,batch,hidden 
        '''
        return output,h_n  # 2,batch,hidden


class PlainDecoder(nn.Module):
    def __init__(self, embeddings, opts):
        super(PlainDecoder, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings=t.FloatTensor(embeddings),
                                                   freeze=False)
        self.dropout = nn.Dropout(opts.drop_rate)
        self.gru = nn.GRU(opts.embed_size, opts.hidden_size, num_layers=opts.num_layers, batch_first=True,
                          bidirectional=True, dropout=opts.drop_rate)
        self.fc = nn.Linear(2 * opts.hidden_size, len(opts.TEXT.vocab.itos))
        self.max_len = opts.maxlen

    # target: batch,seq
    # hidden:num_layer * direction, batch, hidden
    def forward(self, target, target_length, context, encoder_outputs=None):

        seq_len = target.shape[1]

        embed = self.embed(target) #embed:batch,seq,hidden * dir
        embed = self.dropout(embed)



        output_hiddens = []
        hidden = context

        for i in range(seq_len):
            output, hidden = self.gru(embed[:,i,:].unsqueeze(dim=1), hidden)
            output_hiddens.append(output)

        output = t.stack(output_hiddens, dim=1).squeeze(2) #output: batch,seq, hidden
        '''
            output:batch,seq,direction*hidden
            h_0:layer*direction,batch,hidden
        '''
        output = self.fc(output) #batch,seq,vocab
        output = F.log_softmax(output, dim= 2)
        return output, None


class PlainSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(PlainSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        #self._liner = nn.Linear(opts.hidden_size, words) #batch,seq_len,hidden -> batch,seq_len,words

    def forward(self, sources, source_length, targets, targets_length):
        output, context = self.encoder(sources, source_length)
        target, atten = self.decoder(targets, targets_length,context)
        # target：seq,batch,direction*hidden
        return target, atten





