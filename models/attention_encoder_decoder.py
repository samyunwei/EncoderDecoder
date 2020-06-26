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
        self._embed = nn.Embedding.from_pretrained(embeddings=t.FloatTensor(embeddings))
        self.net = nn.Sequential(
            nn.Dropout(p = 0.2),
            nn.GRU(opts.embed_size, opts.hidden_size, num_layers=opts.num_layers, batch_first=True,
                   bidirectional=True, dropout= 0.2)
        )

    #text: batch,seq
    #hidden:num_layer * direction, batch, hidden
    def forward(self, text, length):
        '''
             (seq_len, batch, num_directions * hidden_size):
        '''
        text = self._embed(text)
        output, h_n = self.net(text)
        '''
           output: seq,batch,direction*hidden
           h_0: layer*direction,batch,hidden 
        '''
        return output, h_n


class Attention(nn.Module):
    def __init__(self,input_size,attention_size):
        super(Attention, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, attention_size, bias=False),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False))

    '''
        enc_states:batch,seq, 2 * hidden #2 birdirection
        dec_state:batch,2 * hidden
    '''
    def forward(self, enc_states, dec_state):
        """
            enc_states: (batch, seq, hidden)
            dec_state: (batch, hidden)
            """
        # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
        batch, seq, hidden = enc_states.shape
        dec_states = dec_state[:, None, :].repeat(1, seq, 1)

        enc_and_dec_states = t.cat((enc_states, dec_states), dim=2)
        enc_and_dec_states = enc_and_dec_states.float()
        e = self.net(enc_and_dec_states)  # (batch, seq, 1)  batch,word,score
        alpha = F.softmax(t.squeeze(e, dim=2), dim=1)  # 在时间步维度做softmax运算
        alpha = t.unsqueeze(alpha, dim=2) #batch,word,1 ->
        enc_states = enc_states.float()
        return (alpha * enc_states).sum(dim=1) #batch,hidden. context以这个维度返回主要是为了与Y(t-1)保持同一个维度


class AttenDecoder(nn.Module):
    def __init__(self, embeddings, opts, atten, vocab_size):
        super(AttenDecoder, self).__init__()
        self._embed = nn.Embedding.from_pretrained(embeddings=t.FloatTensor(embeddings),
                                                   freeze=False)
        self._rnn = nn.GRU(opts.embed_size, opts.hidden_size, batch_first=False)
        self._fc = nn.Linear(opts.hidden_size, vocab_size)

        self.dropout = nn.Dropout(0.2)


        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(opts.embed_size + opts.hidden_size * 2, opts.hidden_size, num_layers=opts.num_layers, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(2 * opts.hidden_size, vocab_size)  # 2 is bidirection
        #self.atten = Attention(opts.hidden_size * 4 ,300)
        self.atten = atten


    def forward(self, cur_input, y_state, enc_states):
        '''
                    cur_input shape: batch
                    y_state: num_layers, batch,hidden
                    enc_states: batch,seq, hidden
                '''
        # 解码器在最初时间步的输入是BOS
        batch = enc_states.shape[0]
        if True:  # bidirectional
            y_state_last = y_state[-2:, :, :]  # 2,batch,hidden
            # dec_state:batch,2 * hidden
            y_state_last = y_state_last.contiguous().transpose(0, 1).contiguous().view(batch, -1)

        c = self.atten(enc_states, y_state_last)  # 对于Decoder,只选择最上层的hidden, c：batch,hidden

        # cat c which belong to this moment and cur_input
        print(self._embed(cur_input).shape, c.shape)
        embed = self._embed(cur_input)
        input_and_c = t.cat((embed, c), dim=1)  # input_and_c:batch,hidden

        # input_and_c.unsqueeze(1)——>batch,seq,hidden:batch,1,hidden
        output, state = self.gru(input_and_c.unsqueeze(1), y_state)

        # batch,1,hidden->batch,1,vocab -> batch,vocab
        output = self.fc(output).squeeze(dim=1)

        #
        output = F.log_softmax(output, dim=1)
        # output:batch,vocab
        # state:layer,batch,hidden
        # c:batch,hidden
        return output, state, c






class AttenSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder,device, vocab):
        super(AttenSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.vocab = vocab

    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state

    def forward(self, source, source_length, target, targets_length):

        # enc_output: batch,seq,embed
        # enc_state:layer, batch,hidden
        enc_output, enc_state = self.encoder(source, source_length)
        batch = source.shape[0]
        # num_layers, batch,hidden
        dec_state = self.begin_state(enc_state)
        # 解码器在最初时间步的输入是BOS
        dec_input = t.tensor([ self.vocab.token2id["sos"] ] * batch).to(self.device)

        list_dec_output = []
        list_atten = []

        seq = target.shape[1]

        for index in range(seq):  # Y shape: (batch, seq_len)
            # dec_state: layer,batch,hidden:2*2,32,500
            # dec_input:batch:32
            # enc_output:batch,seq,hidden:32,20,1000
            dec_output, dec_state, c = self.decoder(dec_input, dec_state, enc_output)
            # dec_output：batch,vocab
            list_dec_output.append(dec_output)
            list_atten.append(c)

            dec_input = target[:, index]  # 使用强制教学 逐个输入dec_input

        # target：seq,batch,direction*hidden
        model_output = t.stack(list_dec_output, dim=1)
        attens = t.stack(list_atten, dim=1)
        # model_output:batch,seq,vocab: 32,20,vacab
        # attens:batch,hidden:32,1000
        return model_output, attens



    def translate(self, sources, source_length,  targets, targets_length, maxlen, new_targets, new_target_length, vocab):
        '''
        #context = self._encoder(sources, source_length)
        out, hid = self._encoder(sources, source_length)

        text = t.LongTensor([vocab.token2id["START"]])
        text = t.unsqueeze(text, 0)
        text = text.to(device)
        text_length = t.LongTensor([1]).to(device)

        preds = []
        for index in range(maxlen):
            #output, atten = self._decoder(text, text_length, context)

            output, atten = self._decoder(out, source_length, text, text_length, hid)
            #output: batch,seq,vocab_size
            output = t.max(output,2)[1] #1,seq
            preds.append(t.squeeze(output).item())

            text = output

        answer = [vocab.id2token[item] for item in preds]
        for item in answer:
            if item != "END":
                print(item)
            else:
                break
        '''
        out, hid = self._encoder(sources, source_length)
        preds = []

        batch_size = source_length.shape[0]
        for index in range(batch_size):
            preds = []
            for item_index, data in enumerate(range(maxlen), 0):
                if item_index == 0:
                    # y:batch,seq
                    y = t.unsqueeze(new_targets[index],0)
                else:
                    #temp = t.LongTensor(output.max(2)[1])
                    y = output.max(2)[1]

                output, attns = self._decoder(t.unsqueeze(out[index],0), t.unsqueeze(source_length[index],0), y, t.unsqueeze(new_target_length[index],0), t.unsqueeze(hid[:,index],1))
                preds.append(output.max(2)[1].item())

            answer = [vocab.id2token[item] for item in preds]
            original_answer = [vocab.id2token[item] for item in targets[index].cpu().numpy()]
            print("answer={0},original_answer:{1}".format(answer, original_answer))



        for i in range(maxlen):
            output, atten = self._decoder(out, source_length, sources, source_length, hid)
            y = output.max(2)[1]
            preds.append(y)
        return t.cat(preds, 1)
