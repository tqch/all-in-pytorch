import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=1, input_size=28):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 6, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, input_size//4),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 2, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
