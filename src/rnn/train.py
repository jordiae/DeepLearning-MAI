import torch
import argparse
import os
import logging
import torch.nn as nn
import torch.optim as optim
from rnn.dataset import MathDataset, SortedShufflingDataLoader
import json
from torch.utils.tensorboard import SummaryWriter
from rnn.evaluate import prettify_eval, evaluate
from src.rnn.utils import load_arch


def train(args, train_loader, valid_loader, model, device, optimizer, criterion, logging, resume_info):
    writer = SummaryWriter()
    with open('args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    logging.info(args)
    logging.info(model)
    best_valid_metric = resume_info['best_valid_metric']
    epochs_without_improvement = resume_info['epochs_without_improvement']
    for epoch in range(resume_info['epoch'], args.epochs):
        # train step (full epoch)
        model.train()
        logging.info(f'Epoch {epoch+1} |')
        loss_train = 0.0
        total = 0
        correct = 0
        for idx, data in enumerate(train_loader):
            # if (idx+1) % 10 == 0:
            #     logging.info(f'{idx+1}/{len(train_loader)} batches')
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            total += labels.size(0)
            predicted = torch.round(outputs.data)
            labels = labels.unsqueeze(1).float()
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
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                labels = labels.unsqueeze(1).float()
                loss = criterion(outputs, labels)
                loss_val += loss
                total += labels.size(0)
                predicted = torch.round(outputs.data)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct/total
        logging.info(f'valid: avg_loss = {loss_val/total:.5f} | accuracy = {accuracy:.2f}')
        writer.add_scalar('Avg-loss/valid', loss_val / total, epoch + 1)
        writer.add_scalar('Accuracy/valid', accuracy, epoch + 1)

        torch.save(model.state_dict(), 'checkpoint_last.pt')
        if accuracy > best_valid_metric:
            epochs_without_improvement = 0
            best_valid_metric = accuracy
            torch.save(model.state_dict(), 'checkpoint_best.pt')
            logging.info(f'best valid accuracy: {accuracy:.2f}')
        else:
            epochs_without_improvement += 1
            logging.info(f'best valid accuracy: {best_valid_metric:.2f}')
            if args.early_stop != -1 and epochs_without_improvement == args.early_stop:
                break
        logging.info(f'{epochs_without_improvement} epochs without improvement in validation set')
        with open('resume_info.json', 'w') as f:
            json.dump(resume_info, f, indent=2)

    model = load_arch(None, args, logging)
    model.load_state_dict(torch.load('checkpoint_best.pt'))
    model.to(device)
    eval_res = evaluate(valid_loader, model, device)
    logging.info(prettify_eval('train', *eval_res))


def main():
    # Settings
    parser = argparse.ArgumentParser(description="Train a RNN for Deepmind's Mathematics Dataset")
    parser.add_argument('--arch', type=str, help='Architecture', default='ElmanRNN')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--optimizer', type=str, help='Optimizer', default='Adam')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=32)
    parser.add_argument('--criterion', type=str, help='Criterion', default='bce')
    parser.add_argument('--smooth-criterion', type=float, help='Smoothness for label-smoothing', default=0.1)
    parser.add_argument('--early-stop', type=int,
                        help='Patience in early stop in validation set (-1 -> no early stop)', default=5)
    parser.add_argument('--weight-decay', type=float, help='Weight decay', default=0.001)

    parser.add_argument('--dropout', type=float, help='Dropout in RNN and FC layers', default=0.25)
    parser.add_argument('--embedding-size', type=int, help='Embedding size', default=64)
    parser.add_argument('--hidden-size', type=int, help='Hidden state size', default=128)
    parser.add_argument('--n-layers', type=int, help='Number of recurrent layers', default=1)
    parser.add_argument('--bidrectional', action='store_true', help='Use bidirectional RNNs')
    args = parser.parse_args()

    # Logger
    log_path = 'train.log'
    if os.path.exists('checkpoint_last.pt'):
        logging.basicConfig(filename=log_path, level=logging.INFO, filemode='a')
    else:
        logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    # Load train and validation datasets
    logging.info('===> Loading datasets')
    data_path = os.path.join('..', '..', 'data', 'mathematics', 'mathematics_dataset-v1.0',
                             'train_easy_true_false_concat_subsampled.txt')
    train_dataset = MathDataset(path=data_path, subset='train', sort=True)
    token2idx, idx2token, unk_token_idx = train_dataset.get_vocab()
    vocab_size = len(token2idx)
    valid_dataset = MathDataset(path=data_path, subset='valid', token2idx=token2idx, idx2token=idx2token)
    train_loader = SortedShufflingDataLoader(train_dataset, mode='strict_shuffle', batch_size=args.batch_size)
    valid_loader = SortedShufflingDataLoader(valid_dataset, mode='no_shuffle', batch_size=args.batch_size)

    # Model
    logging.info('===> Building model')
    logging.info(args)

    model = load_arch(vocab_size, args, logging)

    resume_info = dict(epoch=0, best_valid_metric=0.0, epochs_without_improvement=0)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        logging.error("Optimizer not implemented")
        raise NotImplementedError()

    if args.criterion == 'bce':
        criterion = nn.BCELoss()
    else:
        logging.error("Criterion not implemented")
        raise NotImplementedError()

    device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info('===> Training')
    train(args, train_loader, valid_loader, model, device, optimizer, criterion, logging, resume_info)


if __name__ == '__main__':
    main()
