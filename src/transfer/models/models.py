from torch import nn
import torch
import torch.nn.functional as F
import os
import pathlib
import torchvision
from typing import Tuple
from itertools import chain
from typing import Optional
from typing import Callable
from typing import Iterable

SCRIPT_PATH = os.path.join(pathlib.Path(__file__).parent.absolute())


class LinearClassifier(nn.Module):
    def __init__(self, dims_in: int, dims_out: int):
        super().__init__()
        self.fc = nn.Linear(dims_in, dims_out)

    def forward(self, x: torch.Tensor):
        return self.fc(x)


class InitialConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(9, 3, 3, padding=1)
        self.batchnorm2 = self.nn.BatchNorm2d(3)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))
        x = self.conv2(x)
        x = F.relu(self.batchnorm2(x))
        return x


class TransferModel(nn.Module):
    def __init__(self, preconv: Optional[nn.Module], model: nn.Module, get_last_layer: Callable,
                 transfer_strategy: str):
        super().__init__()
        self.preconv = preconv
        self.model = model
        self.get_last_layer = get_last_layer
        self.transfer_strategy = transfer_strategy
        if self.transfer_strategy == 'feature-extraction':
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.get_last_layer(model).parameters():
                param.requires_grad = True

    def forward(self, x):
        if self.preconv is None:
            return self.model(x)
        x = self.preconv(x)
        return self.model(x)

    def get_trainable_parameters(self) -> Iterable[nn.Parameter]:
        if self.preconv is not None and self.transfer_strategy == 'fine-tuning':
            return chain(self.preconv.parameters(), self.model.parameters())
        elif self.preconv is not None and self.transfer_strategy == 'feature-extraction':
            return chain(self.preconv.parameters(), self.get_last_layer(self.model.parameters()))
        else:
            return self.get_last_layer(self.model.parameters())

    def get_pretrained_parameters(self) -> Iterable[nn.Parameter]:
        return self.model.parameters()

    def get_new_parameters(self) -> Iterable[nn.Parameter]:
        if self.preconv is None:
            return self.get_last_layer(self.model.parameters())
        else:
            return chain(self.preconv.parameters(), self.get_last_layer(self.model.parameters()))

    def train(self, *args, **kwargs):
        if self.preconv is not None:
            self.preconv.train(*args, **kwargs)
        if self.transfer_strategy == 'fine-tuning':
            self.model.train(*args, **kwargs)
        else:
            self.get_last_layer(self.model).train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        if self.preconv is not None:
            self.preconv.eval(*args, **kwargs)
        self.model.eval(*args, **kwargs)


def build_pretrained(pretrained_model: str, pretrained: bool, n_classes: int, input_size: Tuple[int, int],
                     transfer_strategy: str, preconv: bool):
    current_dir = os.getcwd()
    os.chdir(SCRIPT_PATH)
    os.chdir(os.path.join('..', '..', '..'))
    torch.hub.set_dir('pretrained_models')
    if pretrained_model == 'resnet-18-imagenet':
        pretrained_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=pretrained)
        pretrained_model.fc = LinearClassifier(512, n_classes)
        transform_in = None  # torchvision.transforms.Resize(input_size)

        def get_last_layer_resnet(resnet):
            return resnet.fc

        get_last_layer = get_last_layer_resnet
    else:
        raise NotImplementedError(f'Pretrained model: {pretrained_model}')
    os.chdir(current_dir)
    model = TransferModel(InitialConv if preconv else None, pretrained_model, get_last_layer, transfer_strategy)
    return model, transform_in


def download_models():
    current_dir = os.getcwd()
    os.chdir(SCRIPT_PATH)
    os.chdir(os.path.join('..', '..', '..'))
    os.makedirs('pretrained_models')
    torch.hub.set_dir('pretrained_models')
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    os.chdir(current_dir)


if __name__ == '__main__':
    download_models()
