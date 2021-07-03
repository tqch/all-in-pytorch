import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from collections import deque


class RowLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, input_size, learnable_initial_state=True):
        super(RowLSTM, self).__init__()
        self.input_state = nn.Conv1d(in_channels, 4*out_channels, 3, 1, 1)
        self.state_state = nn.Conv1d(out_channels, 4*out_channels, 3, 1, 1)
        if learnable_initial_state:
            self.initial_hidden_state = nn.Parameter(
                nn.init.kaiming_normal_(torch.empty(out_channels, input_size[1])))
            self.initial_cell_state = nn.Parameter(
                nn.init.kaiming_normal_(torch.empty(out_channels, input_size[1])))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.resample = in_channels != out_channels
        if self.resample:
            self.project = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.learnable_initial_state = learnable_initial_state

    def unroll(self, single_row, prev_hidden_state=None, prev_cell_state=None):
        self.input_state.weight[:, :, 2].zero_()  # apply mask
        from_input = self.input_state(single_row)

        # initialize recurrent state
        if prev_hidden_state is None:
            if self.learnable_initial_state:
                prev_hidden_state = self.initial_hidden_state.unsqueeze(0).repeat((single_row.size(0), 1, 1))
            else:
                prev_hidden_state = torch.zeros(
                    single_row.size(0), self.out_channels, self.input_size[1]).to(single_row)
        if prev_cell_state is None:
            if self.learnable_initial_state:
                prev_cell_state = self.initial_cell_state.unsqueeze(0).repeat((single_row.size(0), 1, 1))
            else:
                prev_cell_state = torch.zeros(
                    single_row.size(0), self.out_channels, self.input_size[1]).to(single_row)

        from_state = self.state_state(prev_hidden_state)
        gates = from_input + from_state
        gate_sigmoid, content_gate = (
            torch.sigmoid(gates[:, :3 * self.out_channels, :]),
            torch.tanh(gates[:, 3 * self.out_channels:, :])
        )
        output_gate, forget_gate, input_gate = gate_sigmoid.chunk(3, dim=1)
        cell_state = forget_gate * prev_cell_state + input_gate * content_gate
        hidden_state = output_gate * torch.tanh(cell_state)
        return hidden_state, cell_state

    def forward(self, x):
        if self.resample:
            identity = self.project(x)
        else:
            identity = x
        with torch.no_grad():
            self.input_state.weight[:, :, 2].zero_()  # apply mask
        # enhance parallelization
        from_input = self.input_state(x.permute(0, 2, 1, 3).reshape(-1, self.in_channels, self.input_size[1]))
        out = []
        for i in range(self.input_size[0]):  # height
            if i == 0:
                if self.learnable_initial_state:
                    prev_hidden_state = self.initial_hidden_state.unsqueeze(0).repeat((x.size(0), 1, 1))
                    prev_cell_state = self.initial_cell_state.unsqueeze(0).repeat((x.size(0), 1, 1))
                else:
                    prev_hidden_state = torch.zeros(x.size(0), self.out_channels, self.input_size[1]).to(x)
                    prev_cell_state = torch.zeros(x.size(0), self.out_channels, self.input_size[1]).to(x)
            from_state = self.state_state(prev_hidden_state)
            gates = from_input[i::self.input_size[0], :, :] + from_state
            gate_sigmoid, content_gate = (
                torch.sigmoid(gates[:, :3*self.out_channels, :]),
                torch.tanh(gates[:, 3*self.out_channels:, :])
            )
            output_gate, forget_gate, input_gate = gate_sigmoid.chunk(3, dim=1)
            cell_state = forget_gate * prev_cell_state + input_gate * content_gate
            hidden_state = output_gate * torch.tanh(cell_state)
            out.append(hidden_state)
            prev_hidden_state, prev_cell_state = hidden_state, cell_state
        out = torch.stack(out, dim=2) + identity  # skip connection
        return out


class PixelRNN(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            hidden_dim=16,
            out_activation="sigmoid",
            input_size=(28, 28),
            num_layers=3,
            occlude=14
    ):
        super(PixelRNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.num_layers = num_layers
        self.conv = nn.Conv2d(in_channels, hidden_dim, 7, 1, 3)
        self.conv_mask = self._get_conv_mask()
        self.lstm = nn.Sequential(OrderedDict((
                f"row_lstm_{i+1}",
                RowLSTM(hidden_dim, hidden_dim, input_size)
            ) for i in range(num_layers)
        ))
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0)
        )
        self.out_activation = out_activation
        if out_activation == "sigmoid":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        elif out_activation == "softmax":
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        else:
            raise ValueError("Invalid activation type!")

    def _get_conv_mask(self):
        height, width = self.conv.kernel_size
        assert (height % 2 != 0) and (width % 2 != 0), "please use kernels with odd dimension(s) along each axis"
        conv_mask = torch.zeros(height, width)
        conv_mask[:height//2, :] = 1.0
        conv_mask[height//2, :(width+1)//2] = 1.0
        return conv_mask

    def forward(self, x):
        with torch.no_grad():
            self.conv.weight.mul_(self.conv_mask.to(x.device))  # mask the convolution
        conv = self.conv(x[:, :, :-1, :])
        lstm = self.lstm(conv)
        out = self.classifier(F.relu(lstm))
        if self.out_activation == "softmax":
            targets = (x[:, 0, 1:, :]*(self.out_channels-1)).long()
        else:
            targets = x[:, 0, 1:, :]
            out = out.squeeze(1)
        return self.loss_fn(out, targets)

    def generate(self, x):
        
        # used for occluded image (missing upper part) completion only
        state_queue = deque()
        height, width = x.shape[2:]
        
        # incomplete image in original size with 0 padding
        out = torch.cat([
            x, torch.zeros(x.size(0), 1, self.input_size[0]-height, self.input_size[1]).to(x.device)], dim=2)
        
        with torch.no_grad():
            self.conv.weight.mul_(self.conv_mask.to(x.device))  # masked convolution
            for r in range(self.input_size[0]):
                # generate row by row
                if r >= height:
                    if self.out_activation == "softmax":
                        out[:, :, r, :] = self.classifier(
                                F.relu(single_row).unsqueeze(2)).max(dim=1)[1]/(self.out_channels-1)
                    else:
                        out[:, :, r, :] = torch.sigmoid(self.classifier(F.relu(single_row).unsqueeze(2))).squeeze(2)
                        
                single_row = out[:, :, max(0, r-self.conv.kernel_size[0]//2):r+1, :]  # minimal patch size for convolution
                single_row = self.conv(single_row)[:, :, -1, :]
                
                for i in range(self.num_layers):
                    layer = getattr(self.lstm, f"row_lstm_{i+1}")
                    if r > 0:
                        state_queue.append(layer.unroll(single_row, *state_queue.popleft()))
                    else:
                        state_queue.append(layer.unroll(single_row))

                    # skip_connection
                    if layer.resample:
                        identity = layer.project(single_row.unsqueeze(2)).squeeze(2)
                    else:
                        identity = single_row

                        single_row = state_queue[-1][0] + identity
                        
        return out.cpu()
