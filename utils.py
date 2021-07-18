import json
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math
import torch
import numpy as np
from collections import namedtuple, OrderedDict
from functools import reduce
from PIL import Image
from scipy.io import wavfile
from torch.distributions import Categorical

ImageQuality = namedtuple("ImageQuality", ["reconstruction_loss", "mse", "psnr", "ssim"])


def load_configs(fpath="./configs.json"):
    with open(fpath, "rb") as f:
        configs = json.load(f)
    configs["dataset_path"] = os.path.expanduser(configs["dataset_path"])
    return configs


def save_fig(x, filename, save_folder="./figures"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    b, c, h, w = x.shape
    nrow = math.floor(math.sqrt(b))
    x_grid = make_grid(x, nrow=nrow)
    npimg = x_grid.numpy().transpose(1, 2, 0)
    plt.figure(figsize=(5 * nrow, 5 * b // nrow))
    plt.imshow(npimg)
    plt.axis("off")
    plt.savefig(os.path.join(save_folder, f"{filename}.jpg"), dpi=72)
    plt.close()


def jpg2gif(fout, figure_folder, output_size=(512, 512)):
    figures = [
        os.path.join(figure_folder, f)
        for f in sorted(os.listdir(figure_folder))
        if f.endswith(".jpg")
    ]
    images = list(map(lambda x: Image.open(x).resize(output_size), figures))
    images[0].save(
        fout,
        save_all=True,
        append_images=images,
        format="GIF",
        duration=300,
        loop=0
    )
    return figures


def assess_image_quality(x, y):
    """
    assess the image quality via three commonly used indices
    both images are 4-d array with shape of B,C,H,W and already normalized to [0,1]
    :param x: distorted image batch
    :param y: reference image batch
    :return:
        mse: mean squared error
        psnr: peak signal to noise ratio
        ssim: structural similarity index
    """
    b, c, h, w = x.shape
    mse = np.mean(np.power(x - y, 2), axis=(1, 2, 3))
    psnr = np.mean(10 * np.log10(1 / mse))

    # ssim calculation
    mean_intensity = np.mean(x, axis=(1, 2, 3)), np.mean(y, axis=(1, 2, 3))
    L = 1
    K1, K2 = 0.01, 0.03
    C1, C2 = (L * K1) ** 2, (L * K2) ** 2
    C3 = C2 / 2
    luminance = (2 * np.multiply(*mean_intensity) + C1) / (np.sum(np.power(mean_intensity, 2), axis=0) + C1)
    standard_deviation = np.std(x, ddof=1, axis=(1, 2, 3)), np.std(y, ddof=1, axis=(1, 2, 3))
    contrast = (2 * np.multiply(*standard_deviation) + C2) / (np.sum(np.power(standard_deviation, 2), axis=0) + C2)
    covariance = np.sum(
        np.multiply(
            x - mean_intensity[0][:, None, None, None],
            y - mean_intensity[1][:, None, None, None]
        ), axis=(1, 2, 3)
    ) / (c * h * w - 1)
    structure = (covariance + C3) / (np.multiply(*standard_deviation) + C3)
    alpha, beta, gamma = 1, 1, 1
    ssim = np.mean(np.prod(np.power([luminance, contrast, structure], [[alpha], [beta], [gamma]]), axis=0))
    return np.mean(mse), psnr, ssim


class History:

    def __init__(self, index_type=None):
        if index_type is not None:
            self.index_type = index_type
        else:
            self.index_type = namedtuple("DefaultIndex", ["loss"])
        self.history_dict = OrderedDict.fromkeys(self.create_keys())
        self._init_values()

    def _init_values(self):
        for k in self.history_dict.keys():
            self.history_dict[k] = []  # initialize values with empty lists

    def create_keys(self):
        return reduce(
            lambda x, y: x + y,
            map(lambda x: [x + k for k in self.get_indices()], ["train_", "test_"]),
            []
        )

    def update_history(self, train_epoch_dict, test_epoch_dict):
        for prefix, d in [
            ("train_", train_epoch_dict),
            ("test_", test_epoch_dict)
        ]:
            for k, v in d._asdict().items():
                self.history_dict[prefix + k].append(v)
        return self

    def get_last_epoch(self):
        last_epoch = []
        for k, v in self.history_dict.items():
            last_epoch.append((k, v[-1]))
        return OrderedDict(last_epoch)

    def get_indices(self):
        return self.index_type._fields

    def __repr__(self):
        return repr(self.history_dict)


def save_audio(x, filename, sampling_rate, save_folder="./audios"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    audios = np.array(x)
    for i, aud in enumerate(audios):
        wavfile.write(os.path.join(
            save_folder,
            f"{filename}_{i+1}.wav"
        ), sampling_rate, aud)


class NucleusSampler:
    """
    provides a method proposed in the nucleus sampling paper:
    THE CURIOUS CASE OF NEURAL TEXT DeGENERATION
    https://arxiv.org/pdf/1904.09751.pdf
    assume the shape of probs is in the form of (N, C, D1, D2, ...)
    """

    def __init__(self, temperature=1.0, threshold=0.9):
        assert 0 <= threshold <= 1, "p must be within [0,1]!"
        self.temperature = temperature
        self.threshold = threshold

    def sample(self, probs=None, logits=None):
        assert not ((probs is None) and (logits is None)), "At least one of arguments should be specified!"
        if 0 < self.threshold < 1:
            if probs is None:
                probs = torch.softmax(logits / self.temperature, dim=1)
            ranks = probs.argsort(dim=1, descending=True)
            ranked_probs = probs.gather(dim=1, index=ranks)
            inv_ranks = ranks.argsort(dim=1)
            cpf = torch.cumsum(ranked_probs, dim=1)  # culmulative probability function
            least_gtr_p = ((cpf - 2) * (cpf >= self.threshold)).min(dim=1)[0] + 2
            mask = torch.logical_or(
                cpf <= least_gtr_p.unsqueeze(1),
                torch.isclose(cpf, least_gtr_p.unsqueeze(1))  # in case there is any rounding error
            ).int().gather(1, inv_ranks)
            top_p = probs * mask  # top-p vocabulary
            if top_p.ndim > 2:
                top_p = top_p.permute(*([0, ] + [i for i in range(2, top_p.ndim)] + [1, ]))
            return Categorical(probs=top_p).sample()
        elif self.threshold == 0:
            # greedy search
            return probs.max(dim=1)[1]
        else:
            # temperature sampling
            return Categorical(logits=logits / self.temperature).sample()