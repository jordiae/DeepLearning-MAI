import torch

import argparse
import os
import logging

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from cnn.evaluate import evaluate, prettify_eval
from cnn.dataset import Mit67Dataset
from cnn.models import *


def train(args, train_loader, valid_loader, model, device, optimizer, criterion, logging):
    logging.info(args)
    model.train()
    previous_valid_accuracy = 0.0
    for epoch in range(args.epochs):
        logging.info(f'Epoch {epoch+1}')

        # train step (full epoch)
        running_loss_epoch = 0.0
        running_loss_batch = 0.0
        for idx, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss_batch += loss.item()
            if idx % 10 == 0 and idx != 0:
                logging.info(f'10 batches loss: {running_loss_batch/10}')
                running_loss_epoch += running_loss_batch
                running_loss_batch = 0.0

        logging.info(f'Train loss at the end of the epoch: {running_loss_epoch}')

        # valid step
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct/total
        logging.info(f'Validation accuracy: {accuracy}')
        if previous_valid_accuracy < accuracy:
            break
        else:
            previous_valid_accuracy = accuracy
        # TODO Checkpointing


def main():
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
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help='Early stop in validation set with no patience')

    # TODO: parameterize data augmentation...
    args = parser.parse_args()

    LOG_PATH = 'train.log'
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    logging.info('===> Loading datasets')
    data_path = os.path.join('..', '..', 'data', 'mit67', args.data)

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
             # Mit67Dataset(os.path.join(data_path, 'train'), transform=erase_transform),
             Mit67Dataset(os.path.join(data_path, 'train'), transform=jitter_transform),
             ))
    else:
        train_dataset = Mit67Dataset(os.path.join(data_path, 'train'), transform=transform)

    valid_dataset = Mit67Dataset(os.path.join(data_path, 'valid'), transform=transform)
    # test_dataset = Mit67Dataset(os.path.join(data_path, 'test'), transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    # test accuracy shouldn't be here, better run in evaluate once we have selected the model
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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

    # TODO: this shouldn't be here, only eval final model

    #logging.info('==> Test')
    #test(args, test_loader, model, device, logging)

    logging.info('==> Validation')
    logging.info(prettify_eval(evaluate(valid_loader, model, device)))


if __name__ == '__main__':
    main()
