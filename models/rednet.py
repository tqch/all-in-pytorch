import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualEncoder(nn.Module):

    def __init__(self, in_channels=1):
        super(ResidualEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 6, 2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out = self.pool(out2)
        return out1, out2, out


class ResidualDecoder(nn.Module):

    def __init__(self, in_channels=1, input_size=28):
        super(ResidualDecoder, self).__init__()
        self.deconv2 = nn.ConvTranspose2d(128, 128, input_size//4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.deconv0 = nn.ConvTranspose2d(64, in_channels, 2, 2)

    def forward(self, x, encodings):
        out2 = F.relu(self.deconv2(x)+encodings[2])
        out1 = F.relu(self.deconv1(out2)+encodings[1])
        out = self.deconv0(out1) + encodings[0]
        return out


class REDNet(nn.Module):

    def __init__(self, in_channels=1, input_size=28):
        super(REDNet, self).__init__()
        self.encoder = ResidualEncoder(in_channels)
        self.decoder = ResidualDecoder(in_channels, input_size)

    def forward(self, x):
        encoding1, encoding2, encoding = self.encoder(x)
        out = self.decoder(encoding, [x, encoding1, encoding2])
        return out