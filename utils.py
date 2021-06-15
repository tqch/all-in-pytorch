__all__ = ["load_configs", "ImageQuality", "save_fig", "assess_img_quality", "History"]

import json
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math
import numpy as np
from collections import namedtuple, OrderedDict
from functools import reduce


ImageQuality = namedtuple("ImageQuality", ["reconstruction_loss", "mse", "psnr", "ssim"])


def load_configs(fpath="./configs.json"):
    with open(fpath, "rb") as f:
        configs = json.load(f)
    configs["dataset_path"] = os.path.expanduser(configs["dataset_path"])
    return configs


def save_fig(x, fname, save_folder="./figures"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    b, c, h, w = x.shape
    nrow = math.floor(math.sqrt(b))
    x_grid = make_grid(x, nrow=nrow)
    npimg = x_grid.numpy().transpose(1, 2, 0)
    plt.figure(figsize=(5 * nrow, 5 * b // nrow))
    plt.imshow(npimg)
    plt.axis("off")
    plt.savefig(os.path.join(save_folder, f"{fname}.jpg"), dpi=72)
    plt.close()


def assess_img_quality(x, y):
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

    def __init__(self, type):
        self.type = type
        self.history_dict = OrderedDict.fromkeys(self.create_keys())
        self._init_values()

    def _init_values(self):
        for k in self.history_dict.keys():
            self.history_dict[k] = []  # initialize values with empty lists

    def create_keys(self):
        return reduce(
            lambda x,y: x+y,
            map(lambda x: [x+k for k in self.type._fields], ["train_", "test_"]),
            []
        )

    def update_history(self, train_epoch_dict, test_epoch_dict):
        for prefix, d in [
            ("train_", train_epoch_dict),
            ("test_", test_epoch_dict)
        ]:
            for k, v in d._asdict().items():
                self.history_dict[prefix+k].append(v)
        return self

    def get_last_epoch(self):
        last_epoch = []
        for k, v in self.history_dict.items():
            last_epoch.append((k, v[-1]))
        return OrderedDict(last_epoch)

    def __repr__(self):
        return repr(self.history_dict)
