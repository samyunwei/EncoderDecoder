import torch as t
import torch.nn as nn
import torch.nn.functional as F

class LanguageModeCriterion(nn.Module):
    def __init__(self):
        super(LanguageModeCriterion, self).__init__()

    def forward(self, model_target, text_target, mask):
        batch_max_length = mask.shape[1]
        text_target = text_target[:, :batch_max_length]
        model_target = model_target[:, :batch_max_length, :]
        batch, seq = model_target.shape[0], model_target.shape[1]
        model_target = model_target.contiguous().view(batch * seq, -1)
        model_target = F.log_softmax(model_target, dim=1)

        text_target = text_target.contiguous().view(-1, 1)

        loss = -t.gather(model_target, dim=1, index=text_target).view(batch, -1) * mask
        loss = loss.sum() / mask.sum()

        return loss
