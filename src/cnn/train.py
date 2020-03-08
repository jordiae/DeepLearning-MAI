import torch
import torchvision
from torchvision import transforms
import argparse
from cnn.models import *  # TODO check relative imports
import os
from torch import optim
from torch import nn
import importlib
from cnn.dataset import Mit67Dataset   # TODO check relative imports


def train(args):
    # https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    data_path = os.path.join('..', '..', 'data', 'mit67', args.data)
    #ToTensor: [0, 255] -> [0, 1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = Mit67Dataset(os.path.join(data_path, 'train'), transform=transform)
    valid_dataset = Mit67Dataset(os.path.join(data_path, 'valid'), transform=transform, enc=train_dataset.enc)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False)

    # arch = importlib.import_module(args.arch)
    # model = arch()
    if args.arch == 'BaseCNN':
        model = BaseCNN()
    else:
        raise NotImplementedError()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(0, 3):
        print(f'Epoch {epoch+1}')

        # train step (full epoch)
        running_loss_epoch = 0.0
        running_loss_batch = 0.0
        for idx, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss_batch += loss.item()
            if idx % 10 == 0 and idx != 0:
                print(f'10 batches loss: {running_loss_batch/10}')
                running_loss_epoch += running_loss_batch
                running_loss_batch = 0.0
        print(f'Train loss at the end of the epoch: {running_loss_epoch}')

        # valid step
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valid_loader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct/total
        print(f'Validation accuracy: {accuracy}')
        # TODO F1 score, data augmentation, early stop, checkpointing...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN for mit67')
    parser.add_argument('--arch', type=str, help='Architecture', default='BaseCNN')
    parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
    # TODO: parameterize lr, optimizer, stop criterion, data augmentation...
    args = parser.parse_args()
    train(args)

