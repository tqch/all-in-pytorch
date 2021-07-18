import re
import os
from torch.utils.data import Dataset
from scipy.io import wavfile
import math
import torch
import torch.nn.functional as F
import numpy as np


class MuTransformer:
    """
    provide a mu-law encoder and decoder for audio data
    """

    def __init__(
            self,
            input_bit_depth=16,
            output_bit_depth=8,
            quantize=False
    ):
        self.input_bit_depth = input_bit_depth
        self.output_bit_depth = output_bit_depth
        self.input_levels = 2 ** input_bit_depth  # the range of raw data
        self.output_levels = 2 ** output_bit_depth  # the range of encoded data
        self.quantize = quantize

    def mu_law_encode(self, x):
        # assume the possible values are centered at 0
        x = x / (self.input_levels // 2)  # elements of x now lie within [-1, 1]
        # mu-law transformation
        # note that the transformation does not depend on the base of logarithm
        out = x.sgn() * torch.log(1 + self.output_levels // 2 * x.abs()) / math.log(1 + self.output_levels // 2)
        if self.quantize:
            out = ((out + 1) * (self.output_levels // 2)).type(getattr(torch, f"uint{self.output_bit_depth}"))
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
        "bit_size": 16,
        "sampling_rate": 16000,
        "train": {
            "person_ids": (
                "p226", "p227", "p228", "p230", "p231", "p233", "p236",
                "p239", "p243", "p244", "p250", "p254", "p256", "p258",
                "p259", "p267", "p268", "p269", "p270", "p273", "p274",
                "p276", "p277", "p278", "p279", "p282", "p286", "p287"
            ), "num_persons": 28, "sample_size": 11572
        }, "test": {"person_ids": ("p232", "p257"), "num_persons": 2, "sample_size": 824}
    }

    def __init__(
            self,
            root,
            train=True,
            max_length=16000,
            drop_last=False,
            silence_threshold=False,
            left_padding=None,
            transformer=MuTransformer
    ):
        self.root = root
        self.max_length = max_length
        self.drop_last = drop_last
        self.silence_threshold = silence_threshold
        self.left_padding = left_padding

        self.subset = "train" if train else "test"
        self.person_ids = self.info[self.subset]["person_ids"]
        self.num_persons = self.info[self.subset]["num_persons"]
        self.mapping = dict(zip(self.person_ids, range(self.num_persons)))
        self.sample_size = self.info[self.subset]["sample_size"]
        self.data_folder = os.path.join(root, "TinyVCTK", "processed", self.subset)

        if transformer is not None:
            self.transform = transformer
        else:
            self.transform = None

        if os.path.exists(self.data_folder):
            self.data = []
            for filename in sorted([
                fp for fp in os.listdir(self.data_folder)
                if "audio" in fp
            ], key=lambda x: int(re.search("audio(\d+)", x).group(1))):
                self.data.append(torch.load(os.path.join(self.data_folder, filename)))
            self.data = torch.cat(self.data, dim=0)
            self.targets = torch.load(os.path.join(
                self.data_folder, f"{self.data_folder}/VCTK_{self.subset}.labels"))
            with open(f"{self.data_folder}/VCTK_{self.subset}.indices", "r") as f:
                self.offset = list(map(lambda x: int(x),
                                       f.read().strip().split()))
        else:
            os.makedirs(self.data_folder)
            self._preprocessing_data()

        self.total_cuts = len(self.offset) - 1

    def _preprocessing_data(self):
        raw_folder = os.path.join(self.root, "TinyVCTK", self.subset)
        filenames = sorted(os.listdir(raw_folder))
        #         self.targets = torch.LongTensor(list(map(lambda x: self.mapping[x.split("_")[0]], filenames)))
        self.data = []
        self.targets = []
        self.offset = [0]
        output_unit_size = self.info["bit_size"]
        for filename in filenames:
            fs, data = wavfile.read(os.path.join(raw_folder, filename), mmap=True)
            if self.silence_threshold:
                # remove silence from both the start and the end
                try:
                    data = data[slice(*np.where(
                        np.abs(data) >= math.floor(2 ** (output_unit_size - 1) * self.silence_threshold))[0][[0, -1]])]
                except IndexError:
                    continue
                if self.drop_last:
                    if len(data) < self.max_length:
                        continue
                    else:
                        data = data[:len(data) - len(data) % self.max_length]
                self.data.append(torch.Tensor(data))
                # cut long tracks into multiple non-overlapping pieces of max_length
                self.offset.extend([
                    self.offset[-1] + min(i + self.max_length, len(data))
                    for i in range(0, len(data), self.max_length)
                ])
                self.targets.extend([
                    self.mapping[filename.split("_")[0]]
                    for _ in range(math.ceil(len(data) / self.max_length))
                ])

        self.data = torch.cat(self.data)
        self.targets = torch.LongTensor(self.targets)
        if self.transform is not None:
            output_unit_size = self.transform.output_bit_depth
            self.data = self.transform(self.data)
        filebytes = (output_unit_size // 8) * self.offset[-1]
        # split the data into volumes of size no exceeding 100M
        max_volume_size = 100 * 1024 * 1024
        volume_no = 1
        for i in range(0, filebytes, max_volume_size):
            torch.save(
                self.data[i:i + max_volume_size].clone(),  # subsetting without clone is going to store the whole tensor
                f"{self.data_folder}/VCTK_{self.subset}.audio{volume_no}")
            volume_no += 1
        torch.save(self.targets, f"{self.data_folder}/VCTK_{self.subset}.labels")
        with open(f"{self.data_folder}/VCTK_{self.subset}.indices", "w") as f:
            f.write("\n".join(list(map(lambda x: str(x), self.offset))))

    def __getitem__(self, idx):
        data = self.data[self.offset[idx]:self.offset[idx + 1]]
        if isinstance(data, torch.ByteTensor):
            data = data.type(torch.int)
        if len(data) < self.max_length:
            padding_length = self.max_length - len(data)
            if self.transform is not None and self.transform.quantize:
                data = F.pad(data, (0, padding_length), mode="constant", value=-1)  # special padding value
            else:
                data = F.pad(data, (0, padding_length))  # right padding
        if self.left_padding is not None:
            if self.transform is not None and self.transform.quantize:
                data = F.pad(data, (self.left_padding, 0), mode="constant", value=-1)  # special padding value
            else:
                data = F.pad(data, (self.left_padding, 0))  # left padding
        return data, F.one_hot(self.targets[idx], self.num_persons)

    def __len__(self):
        return self.total_cuts

    @staticmethod
    def load_default(root, train=True):
        return TinyVCTK(
            root,
            train,
            max_length=16000,
            drop_last=False,
            silence_threshold=0.1,
            left_padding=None,
            transformer=MuTransformer(
                input_bit_depth=16,
                output_bit_depth=8,
                quantize=True
            )
        )
