from rnn.models import *
import torch
from typing import Tuple
import argparse
import logging
from typing import Union


def load_arch(vocab_size: Union[int, None], args: argparse.Namespace, log: logging) -> torch.nn.Module:
    if vocab_size is not None:
        if hasattr(args, 'vocab_size'):
            raise Exception('Vocabulary size already set')
        args.vocab_size = vocab_size
    if args.arch == 'elman':
        model = VanillaRNN(vocab_size=args.vocab_size, embedding_size=args.embedding_size, hidden_dim=args.hidden_dize,
                           n_layers=args.n_layers, mode='elman')
    elif args.arch == 'jordan':
        model = VanillaRNN(vocab_size=args.vocab_size, embedding_size=args.embedding_size, hidden_dim=args.hidden_dize,
                           n_layers=args.n_layers, mode='jordan')
    elif args.arch == 'AlbertRNN':
        model = AlbertRNN(vocab_size=args.vocab_size, embed_size=64, num_output=1, rnn_model='LSTM', use_last=True,
                          hidden_size=64, num_layers=1)
    else:
        log.error("Architecture not implemented")
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

