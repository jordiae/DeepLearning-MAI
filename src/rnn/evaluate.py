import torch
import argparse
import os
import json
import logging
from rnn.utils import load_arch
from rnn.dataset import MathDataset, SortedShufflingDataLoader
from typing import List
from torch import nn
from typing import Tuple
from typing import Dict
from rnn.utils import init_eval_logging
from typing import Union
import random

class ArgsStruct:
    def __init__(self, **entries):
        """
        Helper class to load class from dictionary
        :param entries: Dictionary
        """
        self.__dict__.update(entries)


def prettify_eval(set_: str, accuracy: float, correct: int, avg_loss: float, n_instances: int,
                  stats: Dict[str, List[int]]):
    """Returns string with prettified classification results"""
    table = 'problem_type accuracy\n'
    for k in sorted(stats.keys()):
        accuracy_ = stats[k][0]/stats[k][1]
        accuracy_ = accuracy_*100
        table += k
        table += ' '
        table += '{:.2f}%\n'.format(accuracy_)

    return '\n' + set_ + ' set average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        avg_loss, correct, n_instances, accuracy) + table + '\n'


def evaluate(data_loader: SortedShufflingDataLoader, encoder: torch.nn.Module, decoder: torch.nn.Module,
             vocab_size: int, device: torch.device, dataset: Union[MathDataset, None] = None,
             verbose: bool = False) -> Tuple[float, int, float, int, Dict[str, List[int]]]:
    """Evaluated a model with the given data loader"""
    if verbose and dataset is None:
        raise RuntimeError('Verbose set to True, but no dataset!')
    correct = 0
    total = 0
    total_loss = 0
    stats = {}
    encoder.eval()
    decoder.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            #if (idx + 1) % 10 == 0:
                #logging.info(f'{idx+1}/{len(data_loader)} batches')
            src_tokens, tgt_tokens, src_lengths, tgt_lengths, problem_types = data[0].to(device), data[1].to(device), \
                                                                              data[2].to(device), data[3].to(device),\
                                                                              data[4]

            encoder_x, encoder_hidden, encoder_cell = encoder(src_tokens, src_lengths)
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell

            transposed_tgt_tokens = tgt_tokens.clone().t()
            # Assuming <BOS> and <EOS> already present in tgt_tokens
            # For the loss and accuracy, we take into account <EOS>, but not <BOS>
            batch_correct = torch.zeros(src_tokens.shape[0]).to(device).long()
            transposed_lengths = torch.ones(src_tokens.shape[0]).long().to(device)
            decoder_x = torch.zeros(src_tokens.shape[0], vocab_size).to(device)
            first = True
            # In inference, we don't apply Teacher forcing
            # Greedy inference (TODO: If we have time, beam search)
            # For efficiency, we don't wait until the output sentence is complete (<EOS>).
            if verbose:
                outputs = []
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

                batch_correct[non_zero_idx] += torch.eq(torch.argmax(decoder_x, dim=1),
                                                        transposed_tgt_tokens[tgt_idx + 1])[non_zero_idx]

                total_loss += criterion(decoder_x[non_zero_idx], transposed_tgt_tokens[tgt_idx + 1][non_zero_idx])

                last_predictions[non_zero_idx] = \
                    torch.argmax(decoder_x, dim=1)[non_zero_idx].view(non_zero_idx.shape[0], 1)

                if verbose:
                    outputs.append(torch.argmax(decoder_x, dim=1))
                if tgt_idx == transposed_tgt_tokens.shape[0] - 2:  # <EOS>
                    break

            outputs = list(map(list, zip(*outputs)))
            # Binary evaluation: either correct (exactly equal, character by character) or incorrect
            for tgt_idx, c in enumerate(batch_correct):
                if problem_types[tgt_idx] not in stats:
                    stats[problem_types[tgt_idx]] = [0, 1]
                if c == tgt_lengths[tgt_idx] - 1:  # Don't consider <BOS>
                    correct += 1
                    stats[problem_types[tgt_idx]][0] += 1
                stats[problem_types[tgt_idx]][1] += 1
            total += tgt_tokens.size(0)

            if verbose:
                for tgt_idx, output in enumerate(outputs):
                    if random.randrange(50) == 1:
                        print(problem_types[tgt_idx])
                        print(f'Question: {dataset.decode(src_tokens[tgt_idx])}')
                        print(f'Hypothesis: {dataset.decode(outputs[tgt_idx])}')
                        print(f'Target: {dataset.decode(tgt_tokens[tgt_idx])}')
                        print('-------')

    accuracy = 100 * correct / total
    return accuracy, correct, total_loss/total, total, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a RNN for Deepmind's Mathematics dataset")
    parser.add_argument('--arch', type=str, help='Architecture')
    parser.add_argument('--models-path', type=str, help='Path to model directory. If more than one path is provided, an'
                                                        'ensemble of models os loaded (not yet implemetned)', nargs='+')
    parser.add_argument('--problem-types', type=str, nargs='*', help='List of problems to load from dataset',
                        default = ['numbers__base_conversion.txt', 'numbers__div_remainder.txt', 'numbers__gcd.txt',
                                   'numbers__is_factor.txt', 'numbers__is_prime.txt', 'numbers__lcm.txt',
                                   'numbers__list_prime_factors.txt', 'numbers__place_value.txt', 'numbers__round_number.txt'])
    parser.add_argument('--dataset-instances', type=int, default=100000,
                        help='Number of total instances we want to load from the dataset')
    parser.add_argument('--decoder-checkpoint', type=str, default='decoder_checkpoint_best.pt',
                        help='Decoder checkpoint name')
    parser.add_argument('--encoder-checkpoint', type=str, default='encoder_checkpoint_best.pt',
                        help='Encoder checkpoint name')
    parser.add_argument('--subset', type=str, help='Data subset', default='test')
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, help='Mini-batch size', default=64)
    parser.add_argument('--verbose', action='store_true', help='Enables verbose mode (prints outputs and targets)')
    args = parser.parse_args()

    init_eval_logging(args.subset)

    data_path = os.path.join('..', '..', 'data', 'mathematics', 'mathematics_dataset-v1.0', 'train-easy')

    device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")

    token2idx = None
    idx2token = None
    for path in args.models_path:
        with open(os.path.join(path, 'args.json'), 'r') as f:
            train_args = ArgsStruct(**json.load(f))
            encoder, decoder = load_arch(device, train_args)
            token2idx = train_args.token2idx if token2idx is None else token2idx
            idx2token = train_args.idx2token if idx2token is None else idx2token
            if train_args.token2idx != token2idx or train_args.idx2token != idx2token:
                raise Exception('Incompatible models')
        encoder.load_state_dict(torch.load(os.path.join(path, args.encoder_checkpoint),
                                           map_location='cpu' if args.no_cuda else None))
        encoder.to(device)

        decoder.load_state_dict(torch.load(os.path.join(path, args.decoder_checkpoint),
                                           map_location='cpu' if args.no_cuda else None))
        decoder.to(device)

        dataset = MathDataset(path=data_path, subset=args.subset, sort=True, token2idx = token2idx, idx2token = idx2token,
                          total_lines=args.dataset_instances, problem_types=args.problem_types)
        #dataset = MathDataset(path=data_path, subset=args.subset, sort=True,
        #                      total_lines=args.dataset_instances, problem_types=args.problem_types)
        data_loader = SortedShufflingDataLoader(dataset, mode='no_shuffle', batch_size=train_args.batch_size)

        print(path)
        eval_res = evaluate(data_loader, encoder, decoder, train_args.vocab_size, device, dataset,
                                verbose=args.verbose)
        logging.info(prettify_eval(args.subset, *eval_res))


