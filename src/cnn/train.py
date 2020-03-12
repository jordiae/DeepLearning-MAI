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
    logging.info(model)
    model.train()
    best_valid_accuracy = 0.0
    for epoch in range(args.epochs):
        # train step (full epoch)
        logging.info(f'epoch {epoch+1}')
        loss_epoch = 0.0
        total_len = len(train_loader)
        correct = 0
        for idx, data in enumerate(train_loader):
            if idx % total_len == int(total_len*0.1):
                logging.info(f'{idx}/{total_len}')
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        accuracy = 100 * correct / total_len
        logging.info(f'train: avg_loss = {loss_epoch/total_len} | accuracy = {accuracy}')

        # valid step
        correct = 0
        total = 0
        loss_val = 0
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_val += loss
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct/total
        logging.info(f'valid: avg_loss = {loss_val/total} | accuracy = {accuracy}')

        torch.save(model.state_dict(), 'checkpoint_last.pt')
        if accuracy > best_valid_accuracy:
            best_valid_accuracy = accuracy
            torch.save(model.state_dict(), 'checkpoint_best.pt')
            logging.info(f'best valid loss: {accuracy}')
        else:
            logging.info(f'best valid loss: {best_valid_accuracy}')
            if args.early_stop:
                break

    logging.info(prettify_eval(evaluate(valid_loader, model, device)))


def main():
    # Settings
    parser = argparse.ArgumentParser(description='Train a CNN for mit67')
    parser.add_argument('--arch', type=str, help='Architecture', default='BaseCNN')
    parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--augment', action='store_true', default=True, help='enables data augmentation')
    parser.add_argument('--optimizer', type=str, help='Optimizer', default='SGD')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=128)
    parser.add_argument('--criterion', type=str, help='Criterion', default='cross-entropy')  # TODO: label smoothing?
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help='Early stop in validation set with no patience')

    parser.add_argument('--kernel_size', type=int, help='Kernel size', default=3)
    parser.add_argument('--dropout', action='store_true', default=True, help='Enables dropout in FC layers (0.5)')
    parser.add_argument('--batch-norm', action='store_true', default=True, help='Enables batch normalization')
    parser.add_argument('--conv_layers', type=int, help='N convolutional layers in each block', default=2)
    parser.add_argument('--conv_blocks', type=int, help='N convolutional blocks', default=5)
    parser.add_argument('--fc-layers', type=int, help='N fully-connected layers', default=3)

    args = parser.parse_args()

    log_path = 'train.log'
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    logging.info('===> Loading datasets')
    data_path = os.path.join('..', '..', 'data', 'mit67', args.data)

    transform = transforms.Compose(
        [transforms.ToTensor(),  # scale to [0,1] float tensor
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])
    if args.augment:
        #  to randomly pick and apply the transformations
        transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomRotation(45),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter()  # Randomly change the brightness, contrast and saturation

            ]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
         ])

    train_dataset = Mit67Dataset(os.path.join(data_path, 'train'), transform=transform)

    valid_dataset = Mit67Dataset(os.path.join(data_path, 'valid'), transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info('===> Building model')
    logging.info(args)
    if args.arch == 'BaseCNN':
        model = BaseCNN()
    elif args.arch == 'AlexNet':
        model = AlexNet()
    elif args.arch == 'FiveLayerCNN':
        model = FiveLayerCNN()
    elif args.arch == 'PyramidCNN':
        model = PyramidCNN(args)
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


if __name__ == '__main__':
    main()
