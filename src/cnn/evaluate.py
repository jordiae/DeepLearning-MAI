import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
import argparse
from torchvision import transforms
import os
from cnn.dataset import Mit67Dataset
import json
from cnn.utils import load_arch
import logging


class ArgsStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def prettify_eval(set_, accuracy, correct, avg_loss, class_report, n_instances):
    return '\n' + set_ + ' set average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n{}\n'.format(
        avg_loss, correct, n_instances, accuracy, class_report)


def evaluate(data_loader, model, device):
    model.eval()
    avg_loss = 0
    correct = 0
    y_output = []
    y_ground_truth = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            avg_loss += F.nll_loss(output, target, reduction='sum').item()  # sum batch loss
            pred = output.argmax(dim=1, keepdim=True)  # index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_output += torch.squeeze(pred).tolist()
            y_ground_truth += torch.squeeze(target).tolist()

        avg_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    class_report = classification_report(y_ground_truth, y_output)
    return accuracy, correct, avg_loss, class_report, len(data_loader.dataset)


if __name__ == '__main__':
    # TODO: test whether it works
    parser = argparse.ArgumentParser(description='Evaluate a CNN for mit67')
    parser.add_argument('--arch', type=str, help='Architecture')
    parser.add_argument('--model-path', type=str, help='Path to model directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_best.pt',  help='Checkpoint name')
    parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
    parser.add_argument('--subset', type=str, help='Data subset', default='test')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=2)
    args = parser.parse_args()
    log_path = f'eval-{args.subset}.log'
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    if args.arch == 'PyramidCNN':
        with open(os.path.join(args.model_path, 'args.json'), 'r') as f:
                train_args = ArgsStruct(**json.load(f))
        model = load_arch(train_args)
    else:
        model = load_arch(None)
    model.load_state_dict(torch.load(os.path.join(args.model_path, args.checkpoint)))
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])
    dataset = Mit67Dataset(os.path.join('..', '..', 'data', 'mit67', args.data, args.subset), transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)
    eval_res = evaluate(data_loader, model, device)
    logging.info(prettify_eval(args.subset, *eval_res))
