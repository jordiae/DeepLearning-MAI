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
from rnn.utils import load_arch, init_train_logging
import numpy as np
from rnn.utils import LabelSmoothingLoss
import time


def train(args, train_loader, valid_loader, encoder, decoder, device, optimizer_encoder, optimizer_decoder, criterion,
          resume_info, dataset, seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    writer = SummaryWriter()
    with open('args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    logging.info(encoder)
    logging.info(decoder)
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) +\
                           sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logging.info(f'Training {total_params} parameters')
    best_valid_metric = resume_info['best_valid_metric']
    epochs_without_improvement = resume_info['epochs_without_improvement']
    t0 = time.time()
    for epoch in range(resume_info['epoch'], args.epochs):
        # train step (full epoch)
        encoder.train()
        decoder.train()
        logging.info(f'Epoch {epoch+1} |')
        loss_train = 0.0
        total = 0
        correct = 0
        for idx, data in enumerate(train_loader):
            if (idx+1) % 10 == 0:
                logging.info(f'{idx+1}/{len(train_loader)} batches')
            src_tokens, tgt_tokens, src_lengths, tgt_lengths = data[0].to(device), data[1].to(device),\
                                                               data[2].to(device), data[3].to(device),
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            encoder_x, encoder_hidden, encoder_cell = encoder(src_tokens, src_lengths)
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell

            loss = 0
            transposed_tgt_tokens = tgt_tokens.t()
            # Assuming <BOS> and <EOS> already present in tgt_tokens
            # For the loss and accuracy, we take into account <EOS>, but not <BOS>
            batch_correct = torch.zeros(src_tokens.shape[0]).to(device).long()
            transposed_lengths = torch.ones(src_tokens.shape[0]).long().to(device)
            decoder_x = torch.zeros(src_tokens.shape[0], args.vocab_size).to(device)
            for tgt_idx, tgt in enumerate(transposed_tgt_tokens):
                tgt = tgt.view(tgt.shape[0], 1)
                future_tgt = transposed_tgt_tokens[tgt_idx+1].view(tgt.shape[0], 1)
                non_zero_idx = (future_tgt != 0).nonzero().t()[0]

                # Teacher forcing
                if decoder_cell is None:
                    decoder_x[non_zero_idx], decoder_hidden[non_zero_idx], _ =\
                        decoder(tgt[non_zero_idx], transposed_lengths[non_zero_idx],
                                decoder_hidden[non_zero_idx].to(device),
                                None)
                else:
                        decoder_x[non_zero_idx], decoder_hidden[non_zero_idx], _ =\
                            decoder(tgt[non_zero_idx], transposed_lengths[non_zero_idx],
                                    decoder_hidden[non_zero_idx],
                                    decoder_cell[non_zero_idx])

                loss += criterion(decoder_x[non_zero_idx], transposed_tgt_tokens[tgt_idx+1][non_zero_idx])

                batch_correct[non_zero_idx] += torch.eq(torch.argmax(decoder_x, dim=1),
                                                        transposed_tgt_tokens[tgt_idx+1])[non_zero_idx]
                if tgt_idx == transposed_tgt_tokens.shape[0]-2:  # <EOS>
                    break

            # Binary evaluation: either correct (exactly equal, character by character) or incorrect
            for tgt_idx, c in enumerate(batch_correct):
                if c == tgt_lengths[tgt_idx] - 1:  # Don't consider <BOS>
                    correct += 1
            total += tgt_tokens.size(0)

            loss.backward()

            # Gradient clipping
            if args.clipping > 0:
                nn.utils.clip_grad_norm_(encoder.parameters(), args.clipping)
                nn.utils.clip_grad_norm_(decoder.parameters(), args.clipping)

            optimizer_encoder.step()
            optimizer_decoder.step()
            loss_train += loss.item()

        accuracy = 100 * correct / total
        logging.info(f'train: avg_loss = {loss_train/total:.5f} | accuracy = {accuracy:.2f}')
        writer.add_scalar('Avg-loss/train', loss_train/total, epoch+1)
        writer.add_scalar('Accuracy/train', accuracy, epoch + 1)

        # valid step
        correct = 0
        total = 0
        loss_val = 0
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for idx, data in enumerate(valid_loader):
                if (idx + 1) % 10 == 0:
                    logging.info(f'{idx+1}/{len(valid_loader)} batches')
                src_tokens, tgt_tokens, src_lengths, tgt_lengths = data[0].to(device), data[1].to(device), \
                                                                   data[2].to(device), data[3].to(device)

                encoder_x, encoder_hidden, encoder_cell = encoder(src_tokens, src_lengths)
                decoder_hidden = encoder_hidden
                decoder_cell = encoder_cell

                transposed_tgt_tokens = tgt_tokens.t()
                # Assuming <BOS> and <EOS> already present in tgt_tokens
                # For the loss and accuracy, we take into account <EOS>, but not <BOS>
                batch_correct = torch.zeros(src_tokens.shape[0]).to(device).long()
                transposed_lengths = torch.ones(src_tokens.shape[0]).long().to(device)
                decoder_x = torch.zeros(src_tokens.shape[0], args.vocab_size).to(device)
                first = True
                # In inference, we don't apply Teacher forcing
                # Greedy inference (TODO: If we have time, beam search)
                # For efficiency, we don't wait until the output sentence is complete (<EOS>).
                for tgt_idx, tgt in enumerate(transposed_tgt_tokens):
                    tgt = tgt.view(tgt.shape[0], 1)
                    if first:
                        last_predictions = tgt
                        first = False
                    future_tgt = transposed_tgt_tokens[tgt_idx + 1].view(tgt.shape[0], 1)
                    non_zero_idx = (future_tgt != 0).nonzero().t()[0]

                    if decoder_cell is None:
                        decoder_x[non_zero_idx], decoder_hidden[non_zero_idx], _ = \
                            decoder(last_predictions[non_zero_idx], transposed_lengths[non_zero_idx],
                                    decoder_hidden[non_zero_idx].to(device),
                                    None)
                    else:
                        decoder_x[non_zero_idx], decoder_hidden[non_zero_idx], _ = \
                            decoder(last_predictions[non_zero_idx], transposed_lengths[non_zero_idx],
                                    decoder_hidden[non_zero_idx],
                                    decoder_cell[non_zero_idx])

                    last_predictions[non_zero_idx] =\
                        torch.argmax(decoder_x, dim=1)[non_zero_idx].view(non_zero_idx.shape[0], 1)
                    loss_val += criterion(decoder_x[non_zero_idx], transposed_tgt_tokens[tgt_idx + 1][non_zero_idx])

                    batch_correct[non_zero_idx] += torch.eq(torch.argmax(decoder_x, dim=1),
                                                            transposed_tgt_tokens[tgt_idx + 1])[non_zero_idx]
                    if tgt_idx == transposed_tgt_tokens.shape[0] - 2:  # <EOS>
                        break

                # Binary evaluation: either correct (exactly equal, character by character) or incorrect
                for tgt_idx, c in enumerate(batch_correct):
                    if c == tgt_lengths[tgt_idx] - 1:  # Don't consider <BOS>
                        correct += 1
                total += tgt_tokens.size(0)

        accuracy = 100 * correct/total
        logging.info(f'valid: avg_loss = {loss_val/total:.5f} | accuracy = {accuracy:.2f}')
        writer.add_scalar('Avg-loss/valid', loss_val / total, epoch + 1)
        writer.add_scalar('Accuracy/valid', accuracy, epoch + 1)

        torch.save(encoder.state_dict(), 'encoder_checkpoint_last.pt')
        torch.save(decoder.state_dict(), 'decoder_checkpoint_last.pt')
        if accuracy > best_valid_metric:
            epochs_without_improvement = 0
            best_valid_metric = accuracy
            torch.save(encoder.state_dict(), 'encoder_checkpoint_best.pt')
            torch.save(decoder.state_dict(), 'decoder_checkpoint_best.pt')
            logging.info(f'best valid accuracy: {accuracy:.2f}')
        else:
            epochs_without_improvement += 1
            logging.info(f'best valid accuracy: {best_valid_metric:.2f}')
            if args.early_stop != -1 and epochs_without_improvement == args.early_stop:
                break
        logging.info(f'{epochs_without_improvement} epochs without improvement in validation set')
        with open('resume_info.json', 'w') as f:
            json.dump(resume_info, f, indent=2)

    t1 = time.time()
    logging.info(f'Finished training in {t1-t0:.1f}s')
    encoder, decoder = load_arch(device, args)
    encoder.load_state_dict(torch.load('encoder_checkpoint_best.pt'))
    decoder.load_state_dict(torch.load('decoder_checkpoint_best.pt'))
    encoder.to(device)
    decoder.to(device)
    eval_res = evaluate(valid_loader, encoder, decoder, args.vocab_size, device)
    logging.info(prettify_eval('valid', *eval_res))


def main():
    # Settings
    parser = argparse.ArgumentParser(description="Train a RNN for Deepmind's Mathematics Dataset")
    parser.add_argument('--arch', type=str, help='Architecture', default='elman')
    parser.add_argument('--problem-types', type=str, nargs='*', help='List of problems to load from dataset',
                        default = ['numbers__base_conversion.txt', 'numbers__div_remainder.txt', 'numbers__gcd.txt',
                                   'numbers__is_factor.txt', 'numbers__is_prime.txt', 'numbers__lcm.txt',
                                   'numbers__list_prime_factors.txt', 'numbers__place_value.txt', 'numbers__round_number.txt'])
    parser.add_argument('--dataset-instances', type=int, default=100000,
                        help='Number of total instances we want to load from the dataset')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--optimizer', type=str, help='Optimizer', default='Adam')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=64)
    parser.add_argument('--criterion', type=str, help='Criterion', default='xent')
    parser.add_argument('--smooth-criterion', type=float, help='Smoothness for label-smoothing', default=0.1)
    parser.add_argument('--early-stop', type=int,
                        help='Patience in early stop in validation set (-1 -> no early stop)', default=10)
    parser.add_argument('--weight-decay', type=float, help='Weight decay', default=0.0001)
    parser.add_argument('--dropout', type=float, help='Dropout in RNN and FC layers', default=0.15)
    parser.add_argument('--embedding-size', type=int, help='Embedding size', default=64)
    parser.add_argument('--hidden-size', type=int, help='Hidden state size', default=128)
    parser.add_argument('--n-layers', type=int, help='Number of recurrent layers', default=1)
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional RNN in the encoder')
    parser.add_argument('--clipping', type=float, help='Gradient clipping', default=0.25)
    parser.add_argument('--share-embeddings', action='store_true', help='Share input and output embeddings')
    parser.add_argument('--no-pytorch-rnn', action='store_true', help='Use our hand-written RNN implementations'
                                                                      '(considerably less efficient) instead of the'
                                                                      'PyTorch ones PyTorch RNN implementations'
                                                                      'instead of our hand-written ones, for'
                                                                      'efficiency.')
    args = parser.parse_args()
    init_train_logging()

    # Load train and validation datasets
    logging.info('===> Loading datasets')
    data_path = os.path.join('..', '..', 'data', 'mathematics', 'mathematics_dataset-v1.0', 'train-easy')
    train_dataset = MathDataset(path=data_path, subset='train', sort=True, total_lines=args.dataset_instances,
                                problem_types=args.problem_types)
    token2idx, idx2token, unk_token_idx = train_dataset.get_vocab()
    vocab_size = len(token2idx)
    args.vocab_size = vocab_size
    args.token2idx = token2idx
    args.idx2token = idx2token
    args.unk_token_idx = unk_token_idx
    valid_dataset = MathDataset(path=data_path, subset='valid', sort=True, token2idx=token2idx, idx2token=idx2token,
                                total_lines=args.dataset_instances, problem_types=args.problem_types)
    train_loader = SortedShufflingDataLoader(train_dataset, mode='strict_shuffle', batch_size=args.batch_size)
    valid_loader = SortedShufflingDataLoader(valid_dataset, mode='no_shuffle', batch_size=args.batch_size)

    # Model
    logging.info('===> Building model')
    logging.info(args)

    device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
    encoder, decoder = load_arch(device, args)
    encoder.to(device)
    decoder.to(device)

    resume_info = dict(epoch=0, best_valid_metric=0.0, epochs_without_improvement=0)

    if args.optimizer == 'SGD':
        optimizer_encoder = optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
        optimizer_decoder = optim.SGD(decoder.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_decoder = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        logging.error("Optimizer not implemented")
        raise NotImplementedError()

    if args.criterion == 'xent':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'label-smoothed-xent':
        criterion = LabelSmoothingLoss(smoothing=args.smooth_criterion)
    else:
        logging.error("Criterion not implemented")
        raise NotImplementedError()

    logging.info('===> Training')
    train(args, train_loader, valid_loader, encoder, decoder, device, optimizer_encoder, optimizer_decoder, criterion,
          resume_info, train_dataset)


if __name__ == '__main__':
    main()
