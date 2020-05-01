import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from typing import List
import torchvision


def dir_path(s: str):
    if os.path.isdir(s):
        return s
    else:
        raise NotADirectoryError(s)


def get_num_pixels(img_path: str):
    width, height = Image.open(img_path).size
    return width, height


def load_model(pretrained_model: str, pre_conv: bool, mode: str, transfer_strategy: str):
    from transfer.models import build_pretrained
    model, transform_in = build_pretrained(pretrained_model, pretrained=mode == 'train', n_classes=67,
                                           input_size=(256, 256), transfer_strategy=transfer_strategy, preconv=pre_conv)
    return model, transform_in


class ComposedOptimizer:
    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        self.optimizers = optimizers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimizers:
            opt.step()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def smooth_one_hot(self, target: torch.Tensor, classes: int, smoothing: float = 0.0):
        assert 0 <= smoothing < 1
        shape = (target.size(0), classes)
        with torch.no_grad():
            target = torch.empty(size=shape, device=target.device) \
                .fill_(smoothing / (classes - 1)) \
                .scatter_(1, target.data.unsqueeze(1), 1. - smoothing)

        return target

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        target = LabelSmoothingLoss.smooth_one_hot(self, target, input.size(-1), self.smoothing)
        lsm = F.log_softmax(input, -1)
        loss = -(target * lsm).sum(-1)
        loss = loss.mean()
        return loss
