from torch.utils.data.dataset import Dataset
import os
from typing import Tuple, List, Dict


class MathDataset(Dataset):
    def __init__(self, path: str, subset: str, token2idx: Dict[str, int] = None, idx2token: List[str] = None,
                 lower: bool = True, props: Tuple[int] = (98, 1, 1)):
        """

        :param path: path to the 'train-easy' dataset of Deepmind's Mathematics dataset (v1.0)
        We use focus on this subset because of computational and time constraints
        :param subset: 'train', 'valid' or 'test' (has implications for vocabulary building)
        :param token2idx: Dictionary mapping each character to the corresponding index
        :param idx2token: Inverse of token2idx, ie. list mapping the corresponding index to the original character
        :param lower: whether to convert characters to lowercase. By default, True, because characters don't seem to add
        information in this dataset
        :param props: proportions of the
        """
        self.path = path
        self.subset = subset
        subsets = ['train', 'valid', 'test']
        self.props = props
        assert self.subset in subsets
        assert token2idx is None if self.subset == 'train' else len(token2idx) > 1
        assert idx2token is None if self.subset == 'train' else len(idx2token) > 1
        assert (token2idx is None and idx2token is None) or len(token2idx) == idx2token
        assert sum(self.props) == 100
        self.unk_token_idx = 0
        self.unk_token = '<UNK>'
        self.token2idx = token2idx if token2idx is not None else {self.token2idx}
        self.idx2token = idx2token if token2idx is not None else [self.unk_token]
        self.proportions = dict(zip(subsets, self.props))
        self.X = []
        self.Y = []
        self.lower = lower

        def check_idx(i: int, sub: str) -> bool:
            """
            Checks whether a given data index belongs to the corresponding subset.
            Notice that we assume that the dataset is already shuffled/randomly generated, which is the case in this
            dataset
            :param i: index to check
            :param sub: subset
            :return: True if the index belongs to the subset, False otherwise
            """
            if sub == 'train' and i % 100 < 98:
                return True
            if sub == 'valid' and i % 100 == 98:
                return True
            if sub == 'test' and i % 100 == 99:
                return True
            return False

        for problem_type in os.listdir(path):
            with open(os.path.join(path, problem_type)), 'r' as f:
                idx_x = 0
                for idx, line in enumerate(f.readlines()):
                    if idx % 2 == 0:
                        if not check_idx(idx, self.subset):
                            idx_x += 1
                            continue
                        x = []
                        for c in line:
                            if c.isalnum():
                                c = c.lower() if self.lower else c
                                if subset == 'train':
                                    self.__add_token(c)
                                x.append(self.token2idx[c])
                        self.X.append(x)
                        idx_x += 1
                    else:
                        if not check_idx(idx, self.subset):
                            idx_x += 1
                            continue
                        y = []
                        for c in line:
                            if c.isalnum():
                                c = c.lower() if self.lower else c
                                if subset == 'train':
                                    self.__add_token(c)
                                y.append(self.token2idx[c])
                        self.Y.append(y)
        self.data_len = len(self.X)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.X[index], self.Y[index]

    def __len__(self) -> int:
        return self.data_len

    def get_vocab(self) -> Tuple[Dict[str, int], List[str], int]:
        """
        Returns the vocabulary of the model (build from the train subset)
        :return: Tuple of token2idx, idx2token, unknown token index in idx2token
        """
        return self.token2idx, self.idx2token, self.unk_token_idx

    def __add_token(self, c: str):
        """
        Adds token to the vocabulary, only for subset = 'train', updating idx2token and token2idx
        :param c: character to add to the vocabulary
        :return:
        """
        assert self.subset == 'train'
        assert len(c) == 1
        if c in self.token2idx:
            return
        self.idx2token.append(c)
        self.token2idx[c] = len(self.idx2token) - 1
