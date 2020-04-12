import torch
from typing import Tuple
import argparse
import os
import logging


def load_arch(args: argparse.Namespace) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Returns initialized Seq2seq model.
    :param args: Arguments from argparse.
    :return: Initialized model
    """
    from rnn.models import VanillaRNN, LSTM, GRU, Decoder
    if args.arch == 'elman':
        encoder = VanillaRNN(vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                             hidden_features=args.hidden_size, n_layers=args.n_layers, mode='elman')
        decoder = Decoder(VanillaRNN(vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                                     hidden_features=args.hidden_size, n_layers=args.n_layers, mode='elman'),
                          args.vocab_size)
    elif args.arch == 'jordan':
        encoder = VanillaRNN(vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                             hidden_features=args.hidden_size, n_layers=args.n_layers, mode='jordan')
        decoder = Decoder(VanillaRNN(vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                                     hidden_features=args.hidden_size, n_layers=args.n_layers, mode='jordan'),
                          args.vocab_size)
    elif args.arch == 'lstm':
        encoder = LSTM(vocab_size=args.vocab_size, embedding_dim=args.embedding_size, hidden_features=args.hidden_size,
                       n_layers=args.n_layers)
        decoder = Decoder(LSTM(vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                               hidden_features=args.hidden_size, n_layers=args.n_layers), args.vocab_size)
    elif args.arch == 'gru':
        encoder = GRU(vocab_size=args.vocab_size, embedding_dim=args.embedding_size, hidden_features=args.hidden_size,
                      n_layers=args.n_layers)
        decoder = Decoder(GRU(vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                              hidden_features=args.hidden_size, n_layers=args.n_layers), args.vocab_size)
    else:
        raise NotImplementedError()
    return encoder, decoder


def pack_right_padded_seq(seqs: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function for packing a right-padded sequence, inspired by the functionality of
    torch.nn.utils.rnn.pack_padded_sequence.
    Instead of relying on a lengths parameter, it assumes that the sequences are zero-padded.
    The function flattens all sequences into a single sequence, ordered by time-step ([first token of first batch,
    first token of second batch,... last token of last batch] and removes padding. It also returns the effective batch
    size at each iteration, which will be [number of first tokens across batch, number of second tokens...].
    lengths is used to verify that the sequence are ordered by length.
    :param seqs: [batch, right-padded tokens]
    :param lengths: [batch]
    :return: ([packed tokens], [effective batch sizes])
    """
    prev = lengths[0]
    for l in lengths:
        if l < prev:
            raise Exception('Unsorted batch!')
        else:
            prev = l
    effective_batch_sizes = (seqs != 0).sum(dim=0)
    seqs = torch.cat((seqs, torch.zeros(seqs.shape[0], 1).long()), dim=-1)
    seqs = seqs.permute(-1, 0).reshape(seqs.shape[0] * seqs.shape[1])  # [batch, tokens] -> [batch*tokens]
    non_pad_idx = (seqs != 0).nonzero().flatten()
    seqs = seqs[non_pad_idx]
    return seqs, effective_batch_sizes


def init_train_logging():
    """Sets logging such that the output is both saved in a file and output to stdout"""
    log_path = 'train.log'
    if os.path.exists('checkpoint_last.pt'):
        logging.basicConfig(filename=log_path, level=logging.INFO, filemode='a')
    else:
        logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())
