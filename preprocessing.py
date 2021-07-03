import torch
import torch.nn as nn
import numpy as np


class GaussianNoise(nn.Module):

    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        return (x+self.sigma*torch.randn_like(x)).clamp(0, 1)


class GaussianFilter(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            kernel_size,
            stride,
            padding,
            sigma=None
    ):
        super(GaussianFilter, self).__init__(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="reflect",
            groups=in_channels,
            bias=False
        )
        self.sigma = sigma or tuple(map(lambda k: k//2/2, self.kernel_size))
        self._set_weights()

    def _set_weights(self):
        radius = tuple(map(lambda k: k//2, self.kernel_size))
        x, y = np.meshgrid(range(-radius[1], radius[1]+1), range(radius[0], -radius[0]-1, -1))
        sampled_weights = 1/(2*np.pi*np.prod(self.sigma))*np.exp(-x**2/(2*self.sigma[1])-y**2/(2*self.sigma[0]))
        normalized = sampled_weights/np.sum(sampled_weights)
        self.weight.data = torch.FloatTensor(np.expand_dims(normalized, (0, 1)).repeat(self.in_channels, axis=0))

    def __call__(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out
