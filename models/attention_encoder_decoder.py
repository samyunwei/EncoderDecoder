import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .no_attention_encoder_decoder import PlainEncoder

'''
    out: batch,seq_len,num_directions * hidden_size
    hid:num_layers * num_directions, batch, hidden_size
'''

#embed=1000, layer=4,hidden=1000

class Encoder(nn.Module):
    def __init__(self, embeddings, opts):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings=t.FloatTensor(embeddings),
                                                  freeze=False)
        self.net = nn.Sequential(
            nn.Dropout(opts.drop_rate),
            nn.GRU(opts.embed_size, opts.hidden_size, num_layers=opts.num_layers, batch_first=True,
                   bidirectional=True, dropout=opts.drop_rate)
        )

    #text: batch,seq
    #hidden:num_layer * direction, batch, hidden
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


class Attention(nn.Module):
    def __init__(self,hidden_size, bidirection, method):
        super(Attention, self).__init__()
        self.method = method

        self.hidden_size = hidden_size
        self.birdiection = bidirection


    '''
        enc_states:batch,seq, 2 * hidden #2 birdirection
        dec_state:batch,2 * hidden
    '''
    def forward(self, last_hidden, lstm_output):
        batch = lstm_output.shape[0]
        hidden = last_hidden.permute((1, 0, 2)).contiguous().view(batch, -1, 1)

        atten = t.einsum("bij,bjk->bi", [lstm_output, hidden])
        alpha = F.softmax(atten, dim=-1).unsqueeze(dim=-1)  # alpha:batch,seq,1
        context1 = t.einsum("bsh,bsi->bh", [lstm_output, alpha])

        context1 = context1.contiguous().view(self.birdiection, batch, self.hidden_size)
        return context1


class AttenDecoder(nn.Module):
    def __init__(self, embeddings, opts, atten):
        super(AttenDecoder, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings=t.FloatTensor(embeddings),
                                                  freeze=False)
        self.dropout = nn.Dropout(opts.drop_rate)
        self.gru = nn.GRU(opts.embed_size, opts.hidden_size, num_layers=opts.num_layers, batch_first=True,
                          bidirectional=True, dropout=opts.drop_rate)

        self.concat = nn.Linear(opts.hidden_size * 2, opts.hidden_size)
        self.fc = nn.Linear(2 * opts.hidden_size, len(opts.TEXT.vocab.itos))

        self.max_len = opts.maxlen
        self.atten = atten

    #context：layer * dir, batch,  hidden
    #encoder_outputs:batch, seq, hidden * dir
    def forward(self, target, target_length, context, encoder_outputs=None):
        final_hidden = context
        seq_len = target.shape[1]

        embed = self.embed(target) #embed:batch,seq,hidden * dir
        embed = self.dropout(embed)

        output_hiddens = []
        hidden = final_hidden #batch,hidden * 2

        for i in range(seq_len):

            #output:batch,1,hidden * 2
            #:hidden: layer * dir, batch, hidden
            output, hidden = self.gru(embed[:,i,:].unsqueeze(dim=1), hidden)

            context = self.atten(hidden, encoder_outputs)

            hidden = self.concat( t.cat([context, hidden], dim=2  )  )




            output_hiddens.append(output)

        output = t.stack(output_hiddens, dim=1).squeeze(2) #output: batch,seq, hidden
        '''
            output:batch,seq,direction*hidden
            h_0:layer*direction,batch,hidden
        '''
        output = self.fc(output) #batch,seq,vocab
        output = F.log_softmax(output, dim= 2)
        return output, None






class AttenSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(AttenSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state

    def forward(self, sources, source_length, targets, targets_length):
        output, context = self.encoder(sources, source_length)
        target, atten = self.decoder(targets, targets_length, context, output)
        # target：seq,batch,direction*hidden
        return target, atten



