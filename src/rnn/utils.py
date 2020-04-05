import os
import torch
import torch.nn.functional as F
from torch import nn


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
