import torch
from typing import Tuple
import argparse
import os
import logging


def load_arch(device: str, args: argparse.Namespace) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Returns initialized encoder and decoder, to be used jointly as a Seq2seq model.
    Notice that if bidirectional is set to True, the hidden_size of the decoder will be multiplied by 2.
    :param device: device
    :param args: Arguments from argparse.
    :return: Initialized model
    """
    from rnn.models import VanillaRNN, LSTM, GRU, Decoder
    decoder_bidirectional_mul = 2 if args.bidirectional else 1
    if args.arch == 'elman':
        encoder = VanillaRNN(device=device, vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                             hidden_features=args.hidden_size, n_layers=args.n_layers, mode='elman',
                             dropout=args.dropout, bidirectional=args.bidirectional)
        decoder = Decoder(VanillaRNN(device=device, vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                                     hidden_features=args.hidden_size*decoder_bidirectional_mul, n_layers=args.n_layers,
                                     mode='elman', dropout=args.dropout, bidirectional=False),
                          args.vocab_size)
    elif args.arch == 'jordan':
        encoder = VanillaRNN(device=device, vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                             hidden_features=args.hidden_size, n_layers=args.n_layers, mode='jordan',
                             dropout=args.dropout, bidirectional=args.bidirectional)
        decoder = Decoder(VanillaRNN(device=device, vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                                     hidden_features=args.hidden_size*decoder_bidirectional_mul, n_layers=args.n_layers,
                                     mode='jordan', dropout=args.dropout, bidirectional=False),
                          args.vocab_size)
    elif args.arch == 'lstm':
        encoder = LSTM(device=device, vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                       hidden_features=args.hidden_size, n_layers=args.n_layers, dropout=args.dropout,
                       bidirectional=args.bidirectional)
        decoder = Decoder(LSTM(device=device, vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                               hidden_features=args.hidden_size*decoder_bidirectional_mul, n_layers=args.n_layers,
                               dropout=args.dropout, bidirectional=False), args.vocab_size)
    elif args.arch == 'gru':
        encoder = GRU(device=device, vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                      hidden_features=args.hidden_size, n_layers=args.n_layers, dropout=args.dropout,
                      bidirectional=args.bidirectional)
        decoder = Decoder(GRU(device=device, vocab_size=args.vocab_size, embedding_dim=args.embedding_size,
                              hidden_features=args.hidden_size*decoder_bidirectional_mul, n_layers=args.n_layers,
                              dropout=args.dropout, bidirectional=False), args.vocab_size)
    else:
        raise NotImplementedError()
    return encoder, decoder


def pack_right_padded_seq(seqs: torch.Tensor, lengths: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function for packing a right-padded sequence, inspired by the functionality of
    torch.nn.utils.rnn.pack_padded_sequence.
    Instead of relying on a lengths parameter, it assumes that the sequences are zero-padded.
    The function flattens all sequences into a single sequence, ordered by time-step ([first token of first batch,
    first token of second batch,... last token of last batch] and removes padding. It also returns the effective batch
    size at each iteration, which will be [number of first tokens across batch, number of second tokens...].
    lengths is used to verify that the sequence are ordered by length.
    If the batch is not sorted by increasing lengths, an exception is thrown.
    :param seqs: [batch, right-padded tokens]
    :param lengths: [batch]
    :param device: device
    :return: ([packed tokens], [effective batch sizes])
    """
    prev = lengths[0]
    for l in lengths:
        if l < prev:
            raise Exception('Unsorted batches!')
        else:
            prev = l
    effective_batch_sizes = (seqs != 0).sum(dim=0)
    seqs = torch.cat((seqs, torch.zeros(seqs.shape[0], 1).to(device).long()), dim=-1)
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
