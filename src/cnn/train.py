import torch
import argparse
import os
import logging
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from cnn.evaluate import evaluate, prettify_eval
from cnn.dataset import Mit67Dataset
import json
from torch.utils.tensorboard import SummaryWriter
from cnn.utils import load_arch


def train(args, train_loader, valid_loader, model, device, optimizer, criterion, logging, resume_info):
    writer = SummaryWriter()
    with open('args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    logging.info(args)
    logging.info(model)
    model.train()
    best_valid_accuracy = resume_info['best_valid_accuracy']
    epochs_without_improvement = resume_info['epochs_without_improvement']
    for epoch in range(resume_info['epoch'], args.epochs):
        # train step (full epoch)
        model.train()
        logging.info(f'Epoch {epoch+1} |')
        loss_train = 0.0
        total = 0
        correct = 0
        for idx, data in enumerate(train_loader):
            if (idx+1) % 10 == 0:
                logging.info(f'{idx+1}/{len(train_loader)} batches')
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        accuracy = 100 * correct / total
        logging.info(f'train: avg_loss = {loss_train/total:.5f} | accuracy = {accuracy:.2f}')
        writer.add_scalar('Avg-loss/train', loss_train/total, epoch+1)
        writer.add_scalar('Accuracy/train', accuracy, epoch + 1)

        # valid step
        correct = 0
        total = 0
        loss_val = 0
        model.eval()
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
        logging.info(f'valid: avg_loss = {loss_val/total:.5f} | accuracy = {accuracy:.2f}')
        writer.add_scalar('Avg-loss/valid', loss_val / total, epoch + 1)
        writer.add_scalar('Accuracy/valid', accuracy, epoch + 1)

        torch.save(model.state_dict(), 'checkpoint_last.pt')
        if accuracy > best_valid_accuracy:
            epochs_without_improvement = 0
            best_valid_accuracy = accuracy
            torch.save(model.state_dict(), 'checkpoint_best.pt')
            logging.info(f'best valid accuracy: {accuracy:.2f}')
        else:
            epochs_without_improvement += 1
            logging.info(f'best valid accuracy: {best_valid_accuracy:.2f}')
            if args.early_stop != -1 and epochs_without_improvement == args.early_stop:
                break
        logging.info(f'{epochs_without_improvement} epochs without improvement in validation set')

    model = load_arch(args)
    model.load_state_dict(torch.load('checkpoint_best.pt'))
    model.to(device)
    eval_res = evaluate(valid_loader, model, device)
    logging.info(prettify_eval('train', *eval_res))


def main():
    # Settings
    parser = argparse.ArgumentParser(description='Train a CNN for mit67')
    parser.add_argument('--arch', type=str, help='Architecture', default='PyramidCNN')
    parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--no-augment', action='store_true', help='disables data augmentation')
    parser.add_argument('--optimizer', type=str, help='Optimizer', default='Adam')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=128)
    parser.add_argument('--criterion', type=str, help='Criterion', default='cross-entropy')  # TODO: label smoothing?
    parser.add_argument('--early-stop', type=int,
                        help='Patience in early stop in validation set (-1 -> no early stop)', default=6)
    parser.add_argument('--weight-decay', type=float, help='Weight decay', default=0.001)

    parser.add_argument('--kernel-size', type=int, help='Kernel size', default=3)
    parser.add_argument('--no-dropout', action='store_true', help='disables dropout in FC layers (0.5)')
    parser.add_argument('--no-batch-norm', action='store_true', help='disables batch normalization')
    parser.add_argument('--conv-layers', type=int, help='N convolutional layers in each block', default=2)
    parser.add_argument('--conv-blocks', type=int, help='N convolutional blocks', default=7)
    parser.add_argument('--fc-layers', type=int, help='N fully-connected layers', default=3)
    parser.add_argument('--initial-channels', type=int, help='Channels out in first convolutional layer', default=16)
    parser.add_argument('--no-pool', action='store_true', help='Replace pooling by stride = 2')

    args = parser.parse_args()
    log_path = 'train.log'
    if os.path.exists('checkpoint_last.pt'):
        logging.basicConfig(filename=log_path, level=logging.INFO, filemode='a')
    else:
        logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    logging.info('===> Loading datasets')
    data_path = os.path.join('..', '..', 'data', 'mit67', args.data)

    transform = transforms.Compose(
        [transforms.ToTensor(),  # scale to [0,1] float tensor
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])
    if not args.no_augment:
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
    model = load_arch(args)
    resume_info = dict(epoch=0, best_valid_accuracy=0.0, epochs_without_improvement=0)
    if os.path.exists('checkpoint_last.pt'):
        model.load_state_dict(torch.load('checkpoint_last.pt'))
        resume_info = json.loads(open('resume_info.json', 'r').read())
        logging.info('Resuming training from checkpoint...')
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    train(args, train_loader, valid_loader, model, device, optimizer, criterion, logging, resume_info)


if __name__ == '__main__':
    main()
