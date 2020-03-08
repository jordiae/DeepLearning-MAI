import argparse
import os
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from cnn.train import train
from cnn.eval import test
from cnn.models import *
from cnn.dataset import Mit67Dataset

# Settings
parser = argparse.ArgumentParser(description='Train a CNN for mit67')
parser.add_argument('--arch', type=str, help='Architecture', default='BaseCNN')
parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=3)
parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001)
parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--augment', action='store_true', default=True, help='enables data augmentation')
parser.add_argument('--optimizer', type=str, help='Optimizer', default='SGD')
parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=2)
parser.add_argument('--criterion', type=str, help='Criterion', default='cross-entropy')

# TODO: parameterize data augmentation...
args = parser.parse_args()

time_str = time.strftime("%Y%m%d-%H%M%S")
experiment_name = args.arch.lower() + time_str
LOG_PATH = os.path.join('..', 'experiments', 'cnn', experiment_name + '.log')
logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
logging.getLogger('').addHandler(logging.StreamHandler())


def main():
    logging.info('===> Loading datasets')
    data_path = os.path.join('..', 'data', 'mit67', args.data)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])
    if args.augment:
        # TODO: Refactor and maybe use transforms.RandomChoice or transforms.RandomApply
        #  to randomly pick and apply the transformations

        # Randomly rotate the image
        rotate_transform = transforms.Compose(
            [transforms.RandomRotation(45),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5))])
        # Flip the image
        flip_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=1),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5))])

        # # Randomly selects a part of an image and erase its pixels (NOT WORKING)
        # erase_transform = transforms.Compose(
        #     [transforms.RandomErasing(p=1),
        #      transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5),
        #                           (0.5, 0.5, 0.5))])

        # Randomly change the brightness, contrast and saturation
        jitter_transform = transforms.Compose(
            [transforms.ColorJitter(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5))])

        train_dataset = torch.utils.data.ConcatDataset(
            (Mit67Dataset(os.path.join(data_path, 'train'), transform=transform),
             Mit67Dataset(os.path.join(data_path, 'train'), transform=rotate_transform),
             Mit67Dataset(os.path.join(data_path, 'train'), transform=flip_transform),
             #Mit67Dataset(os.path.join(data_path, 'train'), transform=erase_transform),
             Mit67Dataset(os.path.join(data_path, 'train'), transform=jitter_transform),
             ))
    else:
        train_dataset = Mit67Dataset(os.path.join(data_path, 'train'), transform=transform)

    valid_dataset = Mit67Dataset(os.path.join(data_path, 'valid'), transform=transform)
    test_dataset = Mit67Dataset(os.path.join(data_path, 'test'), transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info('===> Building model')
    logging.info(args)
    if args.arch == 'BaseCNN':
        model = BaseCNN()
    elif args.arch == 'AlexNet':
        model = AlexNet()
    elif args.arch == 'FiveLayerCNN':
        model = FiveLayerCNN()
    else:
        raise NotImplementedError()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        logging.error("Optimizer not implemented")
        raise NotImplementedError()

    if args.criterion == 'cross-entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        logging.error("Criterion not implemented")
        raise NotImplementedError()

    device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info('===> Training')
    train(args, train_loader, valid_loader, model, device, optimizer, criterion, logging)

    logging.info('==> Test')
    test(args, test_loader, model, device, logging)

if __name__ == '__main__':
    main()