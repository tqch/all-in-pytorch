import torch
import torch.nn as nn


class RowLSTM(nn.Module):
    def __init__(self, in_channels=1, input_size=(28, 28)):
        super(RowLSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, )
        self.input_state = nn.Conv