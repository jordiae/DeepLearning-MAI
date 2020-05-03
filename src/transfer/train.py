import torch
import argparse
import os
import logging
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transfer.evaluate import evaluate, prettify_eval
from cnn.dataset import Mit67Dataset
import json
from torch.utils.tensorboard import SummaryWriter
from transfer.utils import load_model
from transfer.utils import LabelSmoothingLoss
import numpy as np
from transfer.utils import ComposedOptimizer
import pathlib


def train(args: argparse.Namespace, train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader, model: nn.Module, device: torch.device,
          optimizer: ComposedOptimizer, criterion: nn.Module, seed: int =42):

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    writer = SummaryWriter()
    with open('args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    logging.info(args)
    logging.info(model)
    best_valid_metric = 0.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
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
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
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
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
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

    model.load_state_dict(torch.load('checkpoint_best.pt'))
    model.to(device)
    eval_res = evaluate(valid_loader, model, device)
    logging.info(prettify_eval('train', *eval_res))


def main():
    # Settings
    parser = argparse.ArgumentParser(description='Train a CNN for mit67, based on a pre-train model')
    parser.add_argument('--from-pretrained', type=str, help='Pre-trained model to learn', default='resnet-18-imagenet')
    parser.add_argument('--transfer-strategy', type=str, help='Transfer learning strategy (fine-tuning,'
                                                              'feature-extraction, or'
                                                              'feature-extraction-freeze-batchnorm-dropout)',
                        default='fine-tuning')
    parser.add_argument('--pre-conv', action='store_true', help='Whether to add a new convolutional layer')
    parser.add_argument('--data', type=str, help='Dataset', default='256x256-split')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=200)
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.0001)
    parser.add_argument('--lr-pretrained', type=float, help='Learning Rate of the pre-trained model (ignored if'
                                                            'transfer-strategy is set to feature extraction')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--no-augment', action='store_true', help='disables data augmentation')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=32)
    parser.add_argument('--criterion', type=str, help='Criterion', default='label-smooth')
    parser.add_argument('--smooth-criterion', type=float, help='Smoothness for label-smoothing', default=0.1)
    parser.add_argument('--early-stop', type=int,
                        help='Patience in early stop in validation set (-1 -> no early stop)', default=10)
    parser.add_argument('--weight-decay', type=float, help='Weight decay', default=0.001)

    args = parser.parse_args()

    if args.lr_pretrained is None:
        args.lr_pretrained = args.lr

    assert args.transfer_strategy in ['fine-tuning', 'feature-extraction',
                                      'feature-extraction-freeze-batchnorm-dropout']

    log_path = 'train.log'

    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())

    logging.info('===> Building model')
    logging.info('Resuming training from pre-trained checkpoint...')
    model, size_transform = load_model(args.from_pretrained, mode='train',
                                       transfer_strategy=args.transfer_strategy, pre_conv=args.pre_conv)

    logging.info('===> Loading datasets')
    script_path = os.path.join(pathlib.Path(__file__).parent.absolute())
    if os.getcwd() == script_path:
        data_path = os.path.join('..', '..', 'data', 'mit67', args.data)
    else:
        data_path = os.path.join('..', '..', '..', 'data', 'mit67', args.data)

    transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])

    if not args.no_augment:
        aug_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.3),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(),
                        A.RandomGamma(),
                    ],
                    p=0.5
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120,
                            sigma=120 * 0.05,
                            alpha_affine=120 * 0.03
                        ),
                        A.GridDistortion(),
                        A.OpticalDistortion(
                            distort_limit=2,
                            shift_limit=0.5
                        ),
                    ],
                    p=0.5
                ),
                A.OneOf(
                    [
                        A.Blur(),
                        A.CoarseDropout(max_holes=5,
                                        max_height=35,
                                        max_width=35)
                    ],
                    p=0.2
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ],
            p=1
        )
    else:
        aug_transform = transform

    if size_transform is not None:
        transform = A.Compose([transform, size_transform, ToTensorV2()])
        aug_transform = A.Compose([aug_transform, size_transform, ToTensorV2()])
    else:
        transform = A.Compose([transform, ToTensorV2()])
        aug_transform = A.Compose([aug_transform, ToTensorV2()])

    train_dataset = Mit67Dataset(os.path.join(data_path, 'train'), transform=aug_transform)
    valid_dataset = Mit67Dataset(os.path.join(data_path, 'valid'), transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    if args.transfer_strategy in ['feature_extraction', 'feature-extraction-freeze-batchnorm-dropout'] or \
            args.lr == args.lr_pretrained:
        optimizer = optim.Adam(model.get_trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer_pretrained = optim.Adam(model.get_pretrained_parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_new = optim.Adam(model.get_new_parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = ComposedOptimizer([optimizer_pretrained, optimizer_new])

    if args.criterion == 'cross-entropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'label-smooth':
        criterion = LabelSmoothingLoss(smoothing=args.smooth_criterion)
    else:
        logging.error("Criterion not implemented")
        raise NotImplementedError()

    device = torch.device('cuda:0' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    model.to(device)

    logging.info('===> Training')
    train(args, train_loader, valid_loader, model, device, optimizer, criterion)


if __name__ == '__main__':
    main()
