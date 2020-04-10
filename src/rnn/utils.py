from rnn.models import *
import torch
from typing import Tuple
import argparse
import os
import logging


def load_arch(args: argparse.Namespace) -> torch.nn.Module:
    if args.arch == 'elman':
        model = VanillaRNN(vocab_size=args.vocab_size, embedding_dim=args.embedding_size, hidden_features=args.hidden_size,
                           n_layers=args.n_layers, mode='elman')
    elif args.arch == 'jordan':
        model = VanillaRNN(vocab_size=args.vocab_size, embedding_dim=args.embedding_size, hidden_features=args.hidden_size,
                           n_layers=args.n_layers, mode='jordan')
    elif args.arch == 'AlbertRNN':
        model = AlbertRNN(vocab_size=args.vocab_size, embed_size=64, num_output=1, rnn_model='LSTM', use_last=True,
                          hidden_size=64, num_layers=1)
    else:
        raise NotImplementedError()
    return model


def pack_right_padded_seq(seqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function for packing a right-padded sequence, imitating the functionality of torch.nn.utils.rnn.pad_sequence.
    The function flattens all sequences into a single sequence, ordered by time-step ([first token of first batch,
    first token of second batch,... last token of last batch] and removes padding. It also returns the effective batch
    size at each iteration, which will be [number of first tokens across batch, number of second tokens...]
    :param seqs: [batch, right-padded tokens]
    :return: ([packed tokens], [effective batch sizes])
    """

    seqs = seqs.permute(-1, 0).reshape(seqs.shape[0] * seqs.shape[1])  # [batch, tokens] -> [batch*tokens]
    pad_idx = (seqs == 0).nonzero().flatten()
    non_pad_idx = (seqs != 0).nonzero().flatten()
    seqs = seqs[non_pad_idx]
    effective_batch_sizes = pad_idx
    return seqs, effective_batch_sizes


def init_train_logging():
    """Sets logging such that the output is both saved in a file and output to stdout"""
    log_path = 'train.log'
    if os.path.exists('checkpoint_last.pt'):
        logging.basicConfig(filename=log_path, level=logging.INFO, filemode='a')
    else:
        logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())
