import os

from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn

from cnn.models import *


def dir_path(s):
    if os.path.isdir(s):
        return s
    else:
        raise NotADirectoryError(s)


def get_num_pixels(img_path):
    width, height = Image.open(img_path).size
    return width, height


def load_arch(args):
    if args.arch == 'PyramidCNN':
        model = PyramidCNN(args)
    else:
        raise NotImplementedError()
    return model


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def smooth_one_hot(self, target, classes, smoothing=0.0):
        assert 0 <= smoothing < 1
        shape = (target.size(0), classes)
        with torch.no_grad():
            target = torch.empty(size=shape, device=target.device) \
                .fill_(smoothing / (classes - 1)) \
                .scatter_(1, target.data.unsqueeze(1), 1. - smoothing)

        return target

    def forward(self, input, target):
        target = LabelSmoothingLoss.smooth_one_hot(self, target, input.size(-1), self.smoothing)
        lsm = F.log_softmax(input, -1)
        loss = -(target * lsm).sum(-1)
        loss = loss.mean()

        return loss
