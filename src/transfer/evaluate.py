import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from cnn.dataset import Mit67Dataset
import json
from transfer.utils import load_model
from transfer.models import TransferModel
import logging
from typing import Iterable
from typing import Any
from typing import Tuple
from typing import Optional


class ArgsStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def prettify_eval(set_: str, accuracy: float, correct: int, avg_loss: float, class_report: str, n_instances: int):
    return '\n' + set_ + ' set average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n{}\n'.format(
        avg_loss, correct, n_instances, accuracy, class_report)


def evaluate(data_loader: torch.utils.data.DataLoader, model: TransferModel, device: torch.device,
             transform: Optional = None):
    model.eval()
    avg_loss = 0
    correct = 0
    y_output = []
    y_ground_truth = []
    with torch.no_grad():
        for data, target in data_loader:
            if transform is not None:
                for idx, e in enumerate(data):
                    data[idx] = transform(e)
            data, target = data.to(device), target.to(device)
            output = model(data)
            avg_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum batch loss
            pred = output.argmax(dim=1, keepdim=True)  # index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_output += torch.squeeze(pred).tolist()
            y_ground_truth += torch.squeeze(target).tolist()

        avg_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    class_report = classification_report(y_ground_truth, y_output)
    return accuracy, correct, avg_loss, class_report, len(data_loader.dataset)


def evaluate_ensemble(data_loader: torch.utils.data.DataLoader, models: Iterable[Tuple[TransferModel, Any]],
                                                                        device: torch.device):
    avg_loss = 0
    correct = 0
    y_output = []
    y_ground_truth = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data_t = models[0][1](data)
            models[0][0].eval()
            output = models[0][0](data_t)
            for model, transform in models[1:]:
                model.eval()
                data_t = transform(data)
                output += model(data_t)

            avg_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum batch loss
            pred = output.argmax(dim=1, keepdim=True)  # index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_output += torch.squeeze(pred).tolist()
            y_ground_truth += torch.squeeze(target).tolist()

        avg_loss = avg_loss / len(data_loader.dataset) / len(models)
    accuracy = 100. * correct / len(data_loader.dataset)
    class_report = classification_report(y_ground_truth, y_output)
    return accuracy, correct, avg_loss, class_report, len(data_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a CNN for mit67')
    parser.add_argument('--arch', type=str, help='Architecture')
    parser.add_argument('--models-path', type=str, help='Path to model directory. If more than one path is provided, an'
                                                        'ensemble of models os loaded', nargs='+')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pt',  help='Checkpoint name')
    parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
    parser.add_argument('--subset', type=str, help='Data subset', default='test')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=2)
    args = parser.parse_args()
    log_path = f'eval-{args.subset}.log'
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    dataset = Mit67Dataset(os.path.join('..', '..', 'data', 'mit67', args.data, args.subset), transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device('cuda:0' if not args.no_cuda and torch.cuda.is_available() else 'cpu')

    models = []
    for path in args.models_path:
        with open(os.path.join(path, 'args.json'), 'r') as f:
            train_args = ArgsStruct(**json.load(f))
        model, transform_in = load_model(train_args.from_pretrained, mode='eval',
                                         transfer_strategy=train_args.transfer_strategy, pre_conv=train_args.pre_conv)
        model.load_state_dict(torch.load(os.path.join(path, args.checkpoint)))
        model.to(device)
        models.append((model, transform_in))

    if len(models) == 1:
        eval_res = evaluate(data_loader, model, device, transform_in)
    else:
        eval_res = evaluate_ensemble(data_loader, models, device)

    logging.info(prettify_eval(args.subset, *eval_res))
