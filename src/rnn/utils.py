import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple


def dir_path(s):
    if os.path.isdir(s):
        return s
    else:
        raise NotADirectoryError(s)


def load_arch(args):
    if args.arch == 'whatever':
        raise NotImplementedError()
    else:
        raise NotImplementedError()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    @staticmethod
    def smooth_one_hot(target, classes, smoothing=0.0):
        assert 0 <= smoothing < 1
        shape = (target.size(0), classes)
        with torch.no_grad():
            target = torch.empty(size=shape, device=target.device) \
                .fill_(smoothing / (classes - 1)) \
                .scatter_(1, target.data.unsqueeze(1), 1. - smoothing)

        return target

    def forward(self, input_, target):
        target = LabelSmoothingLoss.smooth_one_hot(target, input_.size(-1), self.smoothing)
        lsm = F.log_softmax(input_, -1)
        loss = -(target * lsm).sum(-1)
        loss = loss.mean()

        return loss


def pack_right_padded_seq(seqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function for packing a right-padded sequence, imitating the functionality of torch.nn.utils.rnn.pad_sequence.
    The function flattens all sequences into a single sequence, ordered by time-step ([first token of first batch,
    first token of second batch,... last token of last batch] and removes padding. It also returns the effective batch
    size at each iteration, which will be [number of first tokens across batch, number of second tokens...]
    :param seqs: [batch, right-padded tokens]
    :return: ([packed tokens], [effective batch sizes])
    """

    seqs = seqs.permute(-1, 0).reshape(seqs.shape[0] * seqs.shape[1])  # [batch, tokens] -> [batch*tokens]
    pad_idx = (seqs == 0).nonzero().flatten()
    non_pad_idx = (seqs != 0).nonzero().flatten()
    seqs = seqs[non_pad_idx]
    effective_batch_sizes = pad_idx
    return seqs, effective_batch_sizes

