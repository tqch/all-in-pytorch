import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedCausalConv1d(nn.Module):
    def __init__(
            self,
            kernel_size,
            dilate,
            hidden_dim,
            condition="none",
            padding_mode="constant",
            **kwargs
    ):
        super(DilatedCausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilate = dilate
        self.hidden_dim = hidden_dim
        self.left_padding = (kernel_size - 1) * dilate
        # combined two gates: content gate and output gate
        self.conv = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size, dilation=dilate)
        # global|local|non conditioning
        self.condition = condition
        if condition == "global":
            embed_dim = kwargs["embed_dim"]  # usu. one-hot encoding of speaker ids
            self.conditional = nn.Linear(embed_dim, 2 * hidden_dim)
        elif condition == "local":
            # feat_freq = kwargs["feat_freq"]  # sampling frequency (per sec.) of the other time series
            self.conditional = nn.Conv1d(hidden_dim, hidden_dim, 1, 0, 0)
        self.padding_mode = padding_mode
        # learnable padding
        if padding_mode == "learnable":
            self.padding_vector = nn.Parameter(torch.empty(hidden_dim, 1))
            nn.init.kaiming_normal_(self.padding_vector)  # parameter initialization

    def forward(self, x, y=None, automatic_padding=True):
        identity = x
        if automatic_padding:
            if self.padding_mode == "constant":
                if isinstance(automatic_padding, bool):
                    padded = F.pad(x, (self.left_padding, 0))  # notice: the padding starts from the last dim
                elif isinstance(automatic_padding, int):
                    padded = F.pad(x, (automatic_padding, 0))
                else:
                    raise ValueError("Unexpected type of automatic padding!")
            else:
                if isinstance(automatic_padding, bool):
                    padded = torch.cat([self.padding_vector[None, :, :].repeat((x.size(0), 1, self.left_padding)), x],
                                       dim=-1)
                elif isinstance(automatic_padding, int):
                    padded = torch.cat([self.padding_vector[None, :, :].repeat((x.size(0), 1, automatic_padding)), x],
                                       dim=-1)
                else:
                    raise ValueError("Unexpected type of automatic padding!")
        else:
            padded = x
        content, output = self.conv(padded).chunk(2, dim=1)
        if self.condition == "global":
            content_cond, output_cond = self.conditional(y).chunk(2, dim=1)
            content = content + content_cond.unsqueeze(2)  # broadcast over time
            output = output + output_cond.unsqueeze(2)
        content = torch.tanh(content)  # tanh activation
        output = torch.sigmoid(output)  # sigmoid activation
        out = content * output + identity[:, :, -output.size(-1):]  # skip connection
        return out


class WaveNet(nn.Module):
    def __init__(
            self,
            hidden_dim=64,
            output_dim=256,
            layers_per_block=10,
            num_blocks=2,
            kernel_size=2,
            num_voices=28
    ):
        super(WaveNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers_per_block = layers_per_block
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.num_voices = num_voices
        self.feature = nn.Conv1d(1, hidden_dim, 1, 1, 0)
        for i in range(num_blocks):
            self.add_module(f"block_{i}", self._make_block())
        self.classifier = nn.Sequential(
            nn.Conv1d(hidden_dim, output_dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, 1, 1, 0)
        )
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.use_label = None  # indicator attribute

    def _make_block(self):

        class Block(nn.Module):

            def __init__(self, num_layers, kernel_size, hidden_dim, num_voices):
                super(Block, self).__init__()
                """
                take kernel size 2 as example, 
                the ultimate receptive region of 10 stacked layers is 1024 pixel wide
                """
                for i in range(num_layers):
                    self.add_module(
                        f"dilated_causal_cnn_{i}",
                        DilatedCausalConv1d(
                            kernel_size, kernel_size ** i,
                            hidden_dim, condition="global", embed_dim=num_voices)
                    )
                self.num_layers = num_layers

            def forward(self, x, y):
                for i in range(self.num_layers):
                    layer = getattr(self, f"dilated_causal_cnn_{i}")
                    x = layer(x, y)
                return x

        return Block(self.layers_per_block, self.kernel_size, self.hidden_dim, self.num_voices)

    def generate(self, x, y, full_length=32000):
        layers = []
        for block in [getattr(self, f"block_{i}") for i in range(self.num_blocks)]:
            layers.extend([getattr(block, f"dilated_causal_cnn_{j}") for j in range(self.layers_per_block)])
        out = torch.zeros(x.size(0), full_length).to(x.device)
        hidden_repr = torch.zeros(x.size(0), len(layers) + 1, self.hidden_dim, full_length).to(x.device)
        with torch.no_grad():
            out[:, :x.size(1)] = x
            hidden_repr[:, 0, :, :x.size(1)] = self.feature(x.unsqueeze(1))
            for i, layer in enumerate(layers):
                hidden_repr[:, 1 + i, :, :x.size(1)] = layer(hidden_repr[:, i, :, :x.size(1)], y.float())
            if x.size(1) < full_length:
                out[:, x.size(1)] = self.classifier(
                    hidden_repr[:, -1, :, [x.size(1)]]).max(dim=1)[1].float().flatten() / (self.output_dim // 2) - 1
                for i in range(x.size(0), full_length):
                    hidden_repr[:, 0, :, i] = self.feature(out[:, None, [i]]).squeeze(-1)
                    for j, layer in enumerate(layers):
                        hidden_repr[:, 1 + j, :, i] = layer(
                            hidden_repr[:, j, :, max(0, i - layer.left_padding):(i + 1)],
                            y.float(),
                            automatic_padding=max(0, layer.left_padding - i)).squeeze(-1)
                    if not i == full_length - 1:
                        out[:, i + 1] = self.classifier(
                            hidden_repr[:, -1, :, [i]]).max(dim=1)[1].float().flatten() / (self.output_dim // 2) - 1
            else:
                out = self.classifier(hidden_repr[:, -1, :, :]).max(dim=1)[1]
        return out

    def forward(self, x, y):
        y = y.float()
        feat = self.feature(x.unsqueeze(1))  # expand additional representation dim
        for i in range(self.num_blocks):
            block = getattr(self, f"block_{i}")
            if i == 0:
                out = block(feat, y)
            else:
                out = block(out, y)
        out = self.classifier(out)
        return out
