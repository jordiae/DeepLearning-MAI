from torch import nn
import torch
import torch.nn.functional as F
import os
import pathlib
import imp
import torchvision
from typing import Tuple
from itertools import chain
from typing import Optional
from typing import Callable
from typing import Iterable
import albumentations as A
import os
import torchvision.models

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
        self.batchnorm2 = nn.BatchNorm2d(3)

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
        if self.transfer_strategy in ['feature-extraction', 'feature-extraction-freeze-batchnorm-dropout']:
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
        elif self.preconv is not None and self.transfer_strategy in ['feature-extraction',
                                                                     'feature-extraction-freeze-batchnorm-dropout']:
            return chain(self.preconv.parameters(), self.get_last_layer(self.model).parameters())
        else:
            return self.get_last_layer(self.model).parameters()

    def get_pretrained_parameters(self) -> Iterable[nn.Parameter]:
        return self.model.parameters()

    def get_new_parameters(self) -> Iterable[nn.Parameter]:
        if self.preconv is None:
            return self.get_last_layer(self.model).parameters()
        else:
            return chain(self.preconv.parameters(), self.get_last_layer(self.model).parameters())

    def train(self, *args, **kwargs):
        if self.preconv is not None:
            self.preconv.train(*args, **kwargs)
        if self.transfer_strategy in ['fine-tuning', 'feature-extraction-freeze-batchnorm-dropout']:
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
    pretrained_dir = 'pretrained_models'
    torch.hub.set_dir(os.path.abspath(pretrained_dir)) # Beware: https://github.com/pytorch/pytorch/issues/31944
    if pretrained_model == 'resnet-18-imagenet':
        pretrained_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=pretrained)
        pretrained_model.fc = LinearClassifier(512, n_classes)
        transform_in = None

        def get_last_layer_resnet(resnet):
            return resnet.fc

        get_last_layer = get_last_layer_resnet

    elif pretrained_model == 'resnet-50-imagenet':
        pretrained_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=pretrained)
        pretrained_model.fc = LinearClassifier(2048, n_classes)
        transform_in = None

        def get_last_layer_resnet(resnet):
            return resnet.fc

        get_last_layer = get_last_layer_resnet

    elif pretrained_model == 'resnet-18-places':
        places = torch.load(os.path.join(pretrained_dir, 'resnet18_places365.pth.tar'), map_location=torch.device('cpu'))
        model = torchvision.models.__dict__[places['arch']](num_classes=365)
        checkpoint = places
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.fc = LinearClassifier(512, n_classes)
        pretrained_model = model
        transform_in = None

        def get_last_layer_resnet(resnet):
            return resnet.fc

        get_last_layer = get_last_layer_resnet

    elif pretrained_model == 'resnet-50-places':
        places = torch.load(os.path.join(pretrained_dir, 'resnet50_places365.pth.tar'),
                            map_location=torch.device('cpu'))
        model = torchvision.models.__dict__[places['arch']](num_classes=365)
        checkpoint = places
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.fc = LinearClassifier(2048, n_classes)
        pretrained_model = model
        transform_in = None

        def get_last_layer_resnet(resnet):
            return resnet.fc

        get_last_layer = get_last_layer_resnet

    elif pretrained_model == 'diabetic-retinop':
        imp.load_source('model', os.path.join('pretrained_models', 'diabetic_retinop', 'model.py'))
        pretrained_model = torch.load(os.path.join('pretrained_models', 'diabetic_retinop', 'best_model.pth'))
        #pretrained_model.flatten_dim = int(512 * (input_size[0] / 32) * (input_size[1] / 32))
        #pretrained_model.linear1 = nn.Linear(pretrained_model.flatten_dim, 1024)
        pretrained_model.linear2 = LinearClassifier(512, n_classes)
        transform_in = A.Resize(448, 448)

        def get_last_layer_diabetic_retinop(model):
            return model.linear2

        get_last_layer = get_last_layer_diabetic_retinop

    else:
        raise NotImplementedError(f'Pretrained model: {pretrained_model}')
    os.chdir(current_dir)
    model = TransferModel(InitialConv() if preconv else None, pretrained_model, get_last_layer, transfer_strategy)
    return model, transform_in


def download_models():
    current_dir = os.getcwd()
    os.chdir(SCRIPT_PATH)
    os.chdir(os.path.join('..', '..', '..'))
    pretrained_dir = 'pretrained_models'
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)
    torch.hub.set_dir(os.path.abspath(pretrained_dir))  # Beware: https://github.com/pytorch/pytorch/issues/31944
    _ = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    _ = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
    _ = torch.hub.load_state_dict_from_url('http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar',
                                           model_dir=pretrained_dir, map_location=torch.device('cpu'))
    _ = torch.hub.load_state_dict_from_url('http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar',
                                           model_dir=pretrained_dir, map_location=torch.device('cpu'))
    os.chdir(current_dir)


if __name__ == '__main__':
    download_models()
