import torch
import torch.nn.functional as F
import argparse
import os
import json
import logging
from rnn.utils import load_arch
from rnn.dataset import MathDataset, SortedShufflingDataLoader
from typing import List


class ArgsStruct:
    def __init__(self, **entries):
        """
        Helper class to load class from dictionary
        :param entries: Dictionary
        """
        self.__dict__.update(entries)


def prettify_eval(set_: str, accuracy: float, correct: int, avg_loss: float, n_instances: int):
    """Returns string with prettified classification results"""
    return '\n' + set_ + ' set average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, n_instances, accuracy)


def evaluate(data_loader: SortedShufflingDataLoader, model: torch.nn.Module, device: torch.device):
    """Evaluated a model with the given data loader"""
    model.eval()
    avg_loss = 0
    correct = 0
    y_output = []
    y_ground_truth = []
    with torch.no_grad():
        for data, target, lengths in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, lengths)
            avg_loss += F.binary_cross_entropy(output, target, reduction='sum').item()  # sum batch loss
            pred = output.argmax(dim=1, keepdim=True)  # index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_output += torch.squeeze(pred).tolist()
            y_ground_truth += torch.squeeze(target).tolist()

        avg_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy, correct, avg_loss, len(data_loader.dataset)


def evaluate_ensemble(data_loader: SortedShufflingDataLoader, models: List[torch.nn.Module], device: torch.device):
    """Evaluates ensemble of models with the given data loader"""
    avg_loss = 0
    correct = 0
    y_output = []
    y_ground_truth = []
    with torch.no_grad():
        for data, target, lengths in data_loader:
            data, target = data.to(device), target.to(device)
            models[0].eval()
            output = models[0](data, lengths)
            for model in models[1:]:
                model.eval()
                output += model(data)

            avg_loss += F.binary_cross_entropy(output, target, reduction='sum').item()  # sum batch loss
            pred = output.argmax(dim=1, keepdim=True)  # index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_output += torch.squeeze(pred).tolist()
            y_ground_truth += torch.squeeze(target).tolist()

        avg_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy, correct, avg_loss, len(data_loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a RNN for Deepmind's Mathematics dataset")
    parser.add_argument('--arch', type=str, help='Architecture')
    parser.add_argument('--models-path', type=str, help='Path to model directory. If more than one path is provided, an'
                                                        'ensemble of models os loaded', nargs='+')
    parser.add_argument('--problem-types', type=str, nargs='*', help='List of problems to load from dataset')
    parser.add_argument('--dataset-instances', type=int, default=100000,
                        help='Number of total instances we want to load from the dataset')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pt',  help='Checkpoint name')
    parser.add_argument('--subset', type=str, help='Data subset', default='test')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=2)
    args = parser.parse_args()
    log_path = f'eval-{args.subset}.log'
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())
    data_path = os.path.join('..', '..', 'data', 'mathematics', 'mathematics_dataset-v1.0', 'train-easy')

    device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    models = []
    token2idx = None
    idx2token = None
    for path in args.models_path:
        with open(os.path.join(path, 'args.json'), 'r') as f:
            train_args = ArgsStruct(**json.load(f))
            model = load_arch(train_args)
            token2idx = train_args.token2idx if token2idx is None else token2idx
            idx2token = train_args.idx2token if token2idx is None else idx2token
            if train_args.token2idx != token2idx or train_args.idx2token != idx2token:
                raise Exception('Incompatible models')
        model.load_state_dict(torch.load(os.path.join(path, args.checkpoint)))
        model.to(device)
        models.append(model)

    dataset = MathDataset(path=data_path, subset=args.subset, token2idx=token2idx, idx2token=idx2token, sort=True,
                          total_lines=args.dataset_instances, problem_types=args.problem_types)
    data_loader = SortedShufflingDataLoader(dataset, mode='no_shuffle', batch_size=train_args.batch_size)

    if len(models) == 1:
        eval_res = evaluate(data_loader, models[0], device)
    else:
        eval_res = evaluate_ensemble(data_loader, models, device)

    logging.info(prettify_eval(args.subset, *eval_res))
