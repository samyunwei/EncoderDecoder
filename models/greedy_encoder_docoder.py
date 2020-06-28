import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder,device, args):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.TEXT = args.TEXT

    def forward(self, input_seq, input_length, max_length=20):
        # Encoder的Forward计算
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # 把Encoder最后时刻的隐状态作为Decoder的初始值
        decoder_hidden = encoder_hidden
        # 因为我们的函数都是要求(time,batch)，因此即使只有一个数据，也要做出二维的。
        # Decoder的初始输入是SOS
        #decoder_input = t.ones(1, 1, device=self.device, dtype=t.long) * SOS_token
        #Todo 要修改
        decoder_input = t.ones(1, 1, device=self.device, dtype=t.long) * self.TEXT.vocab.stoi["hi"]
        # 用于保存解码结果的tensor
        all_tokens = t.zeros([0], device=self.device, dtype=t.long)
        all_scores = t.zeros([0], device=self.device)
        # 循环，这里只使用长度限制，后面处理的时候把EOS去掉了。
        for _ in range(max_length):
            # Decoder forward一步
            #other
            #decoder_output, decoder_hidden = self.decoder(decoder_input, target_length=None, context=decoder_hidden, encoder_outputs=encoder_outputs)

            decoder_output, decoder_hidden = self.decoder(decoder_input, last_hidden=decoder_hidden,
                                                          encoder_outputs=encoder_outputs)

            # decoder_outputs是(batch=1, vob_size)
            # 使用max返回概率最大的词和得分
            decoder_scores, decoder_input = t.max(decoder_output, dim=1)
            # 把解码结果保存到all_tokens和all_scores里
            all_tokens = t.cat((all_tokens, decoder_input), dim=0)
            all_scores = t.cat((all_scores, decoder_scores), dim=0)
            decoder_input = t.unsqueeze(decoder_input,dim=0)
            # decoder_input是当前时刻输出的词的ID，这是个一维的向量，因为max会减少一维。
            

        # 返回所有的词和得分。
        return all_tokens, all_scores