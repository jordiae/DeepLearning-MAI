from torch.utils.data.dataset import Dataset
import os
from typing import Tuple, List, Dict, Union
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
import torch


class MathDataset(Dataset):
    def __init__(self, path: str, subset: str, token2idx: Dict[str, int] = None, idx2token: List[str] = None,
                 lower: bool = True, props: Tuple[int] = (80, 10, 10), debug: bool = False,
                 sort: Union[bool, str] = 'auto'):
        """

        :param path: path to 'train_easy_true_false_concat_subsampled.txt', our subset from the 'train-easy' dataset of
        Deepmind's Mathematics dataset (v1.0). We use focus on the Train/False questions because of computational and
        time constraints
        :param subset: 'train', 'valid' or 'test' (has implications for vocabulary building)
        :param token2idx: Dictionary mapping each character to the corresponding index
        :param idx2token: Inverse of token2idx, ie. list mapping the corresponding index to the original character
        :param lower: whether to convert characters to lowercase. By default, True, because characters don't seem to add
        information in this dataset
        :param props: proportions of the train-valid-test split
        :param sort: whether to sort the dataset in increasing order. If set to 'auto' (default), it is automatically
        set to True for train and False for valid and test.
        """
        self.path = path
        self.subset = subset
        subsets = ['train', 'valid', 'test']
        self.props = props
        assert self.subset in subsets
        assert token2idx is None if self.subset == 'train' else len(token2idx) > 0
        assert idx2token is None if self.subset == 'train' else len(idx2token) > 0
        assert (token2idx is None and idx2token is None) or len(token2idx) == len(idx2token)
        assert sum(self.props) == 100
        assert sort in ['auto', True, False]
        autosort = True if self.subset == 'train' else False
        self.sort = sort if sort is not 'auto' else autosort
        self.unk_token_idx = 1
        self.unk_token = '<UNK>'
        self.pad_token_idx = 0
        self.pad_token = '<PAD>'
        self.token2idx = token2idx if token2idx is not None else {self.pad_token: 0, self.unk_token: 1}
        self.idx2token = idx2token if token2idx is not None else [self.pad_token, self.unk_token]
        self.proportions = dict(zip(subsets, self.props))
        self.X = []
        self.y = []
        self.lower = lower
        self.debug = debug
        self.lines_debug = 100000
        self.sorted = False
        self.eos_token = '<EOS>'
        self.total_lines = 100000

        def check_idx(i: int, sub: str) -> bool:
            """
            Checks whether a given data index belongs to the corresponding subset.
            Notice that we assume that the dataset is already shuffled/randomly generated, which is the case in this
            dataset
            :param i: index to check
            :param sub: subset
            :return: True if the index belongs to the subset, False otherwise
            """
            if sub == 'train' and i % 100 < self.proportions['train']:
                return True
            if sub == 'valid' and i % 100 < (self.proportions['train'] + self.proportions['valid']):
                return True
            if sub == 'test' and i % 100 >= (self.proportions['train'] + self.proportions['valid']):
                return True
            return False

        self.lengths = []
        with open(os.path.join(path), 'r') as f:
            for idx, line in tqdm(enumerate(f.readlines()), total=self.total_lines*self.proportions[self.subset]//100):
                if not check_idx(idx, self.subset):
                    continue
                if len(line.split()) == 0:
                    break
                input_ = line[:-7]
                label = line[-6:].strip()
                label = 1 if label == 'True' else 0
                x = []
                for c in input_:
                    c = c.lower() if self.lower and c.isalnum() else c
                    if subset == 'train':
                        self.__add_token(c)
                    else:
                        if c not in self.token2idx:
                            c = self.unk_token
                    x.append(self.token2idx[c])
                self.X.append(x)
                self.y.append(label)
                self.lengths.append(len(x))

        assert len(self.X) == len(self.y)
        self.data_len = len(self.X)
        if self.sort:
            self._sort_by_lengths()

    def __getitem__(self, index: int) -> Tuple[List[int], int, int]:
        return self.X[index], self.y[index], self.lengths[index]

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
        assert len(c) == 1 or c == self.eos_token
        if c in self.token2idx:
            return
        self.idx2token.append(c)
        self.token2idx[c] = len(self.idx2token) - 1

    def _sort_by_lengths(self):
        """
        Sorts the dataset in increasing order of total length (X + Y), for efficiency purposes when generating batches
        :return:
        """
        assert len(self.X) > 0
        assert not self.sorted
        sorted_idx = np.argsort(self.lengths)
        self.X = [self.X[i] for i in sorted_idx]
        self.y = [self.y[i] for i in sorted_idx]
        self.lengths = [self.lengths[i] for i in sorted_idx]
        self.sorted = True

    def sort_by_lengths(self):
        """
        Public wrapper for sort_by_lengths (typically, for debugging purposes, if sort is set to False)
        :return:
        """
        self._sort_by_lengths()

    def encode(self, to_encode: str) -> List[int]:
        return list(map(lambda x: self.token2idx[x], to_encode))

    def decode(self, to_decode: List[int]) -> str:
        return ''.join(list(map(lambda x: self.idx2token[x], to_decode)))


def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    _, labels, lengths = zip(*data)
    max_len = max(lengths)
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    return features.float(), labels.long(), lengths.long()


def custom_collate(data):
    # https://discuss.pytorch.org/t/how-to-create-batches-of-a-list-of-varying-dimension-tensors/50773/14
    longest = len(data[-1][0]) + len(data[-1][1])
    for sample_input, sample_output, length in data:
        if len(sample_input) + sample_output < longest:
            pass
    return data


class SortedRandomSampler(RandomSampler):
    def __init__(self, *args, strict: bool, chunks: int = 100, **kwargs):
        super(SortedRandomSampler).__init__(*args, **kwargs)
        self.strict = strict
        self.chunks = chunks
        assert self.chunks > 0

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            raise NotImplementedError()
        if self.strict:
            length_chunks = self.unique_indexes(dataset.lengths)
            chunks = []
            current_chunk = []
            current_length_idx_idx = 1
            current_length_idx = length_chunks[current_length_idx_idx]
            for idx in range(0, n):
                if idx == current_length_idx:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_length_idx_idx += 1
                    current_length_idx = length_chunks[current_length_idx_idx]
                current_chunk.append(idx)
        else:
            chunks = np.array_split(list(range(0, n)), 5)
        res = []
        for chunk in chunks:
            chunk = torch.randperm(chunk[-1]).tolist()
            res.extend(chunk)
        return iter(res)

    @staticmethod
    def unique_indexes(a: List):
        res = []
        already_seen = set([])
        for idx, elem in enumerate(a):
            if elem in already_seen:
                continue
            already_seen.add(elem)
            res.append(idx)
        return res


class VariableSizeDataLoader(DataLoader):
    def __init__(self, dataset: MathDataset, *args, strict: bool, chunks: int = None, **kwargs):
        super(VariableSizeDataLoader).__init__(*args, sampler=SortedRandomSampler(dataset, strict, *args, chunks=chunks),
                                                collate_fn=custom_collate, **kwargs)


if __name__ == '__main__':
    print('Train')
    dataset = MathDataset(path=os.path.join('..', '..', 'data', 'mathematics', 'mathematics_dataset-v1.0',
                                            'train_easy_true_false_concat_subsampled.txt'),
                          subset='train', sort=False)
    print(f"first sequence: {(dataset.decode(dataset.X[0]), True if dataset.y[0] == 1 else False)}")
    print()
    print(f"Encoded first sequence: {(dataset.X[0], dataset.y[0])}")
    print()
    print('Without sorting by length (length of first instances)')
    print(list(map(lambda i: len(i), dataset.X[0:10])))
    print()
    dataset.sort_by_lengths()
    print('Having sorted by length (length first instances)')
    print(list(map(lambda i: len(i), dataset.X[0:10])))
    print()
    token2idx, idx2token, unk_token_idx = dataset.get_vocab()
    print(f'token2idx: len: {len(token2idx)}, examples: {list(token2idx[e] for e in idx2token[0:10])}')
    print(f'idx2token: len: {len(idx2token)}, examples: {idx2token[0:10]}')
    print(f'unk_token_idx: {unk_token_idx}, unk_token: {idx2token[unk_token_idx]}')
    print()
    print('Valid')
    dataset = MathDataset(path=os.path.join('..', '..', 'data', 'mathematics', 'mathematics_dataset-v1.0',
                                            'train_easy_true_false_concat_subsampled.txt'),
                          subset='valid', token2idx=token2idx, idx2token=idx2token)

