import torch
from typing import Tuple
import argparse
import os
import logging
from rnn.models import build_model


def load_arch(args: argparse.Namespace) -> torch.nn.Module:
    model = build_model(args)
    return model


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
