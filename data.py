import os
from torch.utils.data import Dataset
from scipy.io import wavfile
import math
import torch
import torch.nn.functional as F


class MuTransformer:
    """
    provide essential pre/post-processing pipeline for audio data,
    additionally, it will truncate the input data that exceed the max_length
    """
    def __init__(
            self,
            max_length=None,
            input_bit_depth=16,
            output_bit_depth=8
         ):
        self.max_length = max_length
        self.input_bit_depth = input_bit_depth
        self.output_bit_depth = output_bit_depth
        self.input_levels = 2 ** input_bit_depth  # the range of raw data
        self.output_levels = 2 ** output_bit_depth   # the range of encoded data

    def mu_law_encode(self, x):
        # assume the possible values are centered at 0
        x = x / (self.input_levels // 2)  # elements of x now lie within [-1, 1]
        if self.max_length is not None:
            x = x[:self.max_length]  # truncation
            if len(x) < self.max_length:
                x = F.pad(x, (self.max_length-len(x), 0))  # left padding
        # mu-law transformation
        # note that the transformation does not depend on the base of logarithm
        out = x.sgn() * torch.log(1 + self.output_levels // 2 * x.abs()) / math.log(1 + self.output_levels // 2)
        return out

    def mu_law_decode(self, x):
        # assume the possible values are non-negative
        x = x / (self.output_levels // 2) - 1  # elements of x now lie within [-1, 1]
        out = x.sgn() * (torch.exp(math.log(1 + self.output_levels // 2) * x.abs()) - 1)
        out = self.input_levels / self.output_levels * out  # restore the original levels
        return torch.round(out).type(getattr(torch, f"int{self.input_bit_depth}"))  # restore the original input

    def __call__(self, x):
        return self.mu_law_encode(x)


class TinyVCTK(Dataset):
    """
    Contains a subset of original VCTK dataset
    adapted from Device Recorded VCTK (Small subset version)
    audios are collected from 30 persons
    train size: 11572
    test size: 824
    bit size: 16-bit
    sampling rate: 16k Hz
    bit rate: 256kbps
    """
    info = {
        "bit size": 16,
        "sampling rate": 16000,
        "train": {
            "person": (
                "p226", "p227", "p228", "p230", "p231", "p233", "p236",
                "p239", "p243", "p244", "p250", "p254", "p256", "p258",
                "p259", "p267", "p268", "p269", "p270", "p273", "p274",
                "p276", "p277", "p278", "p279", "p282", "p286", "p287"
            ), "num_voices": 28, "sample_size": 11572
        }, "test": {"person": ("p232", "p257"), "num_voices": 2, "sample_size": 824}
    }

    def __init__(
            self,
            root,
            train=True,
            transformer=MuTransformer,
            **transform_kwargs
    ):
        self.root = root
        self.subset = "train" if train else "test"
        self.data_folder = os.path.join(root, "TinyVCTK", self.subset)
        self.data = sorted(os.listdir(self.data_folder))
        self.person_ids = self.info[self.subset]["person"]
        self.num_voices = self.info[self.subset]["num_voices"]
        self.mapping = dict(zip(self.person_ids, range(self.num_voices)))
        self.targets = torch.LongTensor(list(map(lambda x: self.mapping[x.split("_")[0]], self.data)))
        self.sample_size = self.info[self.subset]["sample_size"]
        if transformer is not None:
            self.transform = transformer(**transform_kwargs)
        else:
            self.transform = None

    def __getitem__(self, idx):
        fs, data = wavfile.read(os.path.join(self.data_folder, self.data[idx]), mmap=True)
        x = torch.FloatTensor(data)
        y = self.targets[idx]
        if self.transform is not None:
            return self.transform(x), F.one_hot(y, self.num_voices)
        else:
            return x, F.one_hot(y, self.num_voices)

    def __len__(self):
        return self.sample_size

    @staticmethod
    def load_default(root, train=True):
        return TinyVCTK(
            root,
            train,
            transformer=MuTransformer,
            max_length=32000,
            input_bit_depth=16,
            output_bit_depth=8
        )
