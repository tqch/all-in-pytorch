import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NucleusSampler


class DilatedCausalConv1d(nn.Module):
    def __init__(
            self,
            kernel_size,
            dilate,
            input_dim,
            output_dim,
            skip_dim,
            embed_dim=None,  # usu. one-hot encoding of speaker ids
            condition="none",
            padding_mode="constant"
    ):
        super(DilatedCausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilate = dilate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.left_padding = (kernel_size - 1) * dilate

        # combined two gates: content gate and output gate
        self.conv1 = nn.Conv1d(input_dim, 2 * output_dim, kernel_size, dilation=dilate)
        self.conv2 = nn.Conv1d(output_dim, output_dim, 1, 1, 0)  # depthwise convolution

        # check whether the input_dim matches the output_dim
        self.in_eq_out = input_dim == output_dim

        # residual connection to the end of **this layer**
        self.residual_connection = None
        if not self.in_eq_out:
            self.residual_connection = nn.Conv1d(input_dim, output_dim, 1, 1, 0)

        # global|local|non conditioning
        self.condition = condition
        if condition == "global":
            self.conv3 = nn.Linear(embed_dim, 2 * output_dim, bias=False)

        elif condition == "local":
            # assume the other supplementary low-frequency time series (linguistic features such as text sequences)
            # has already been upsampled to the same resolution as the input audio
            self.conv3 = nn.Conv1d(embed_dim, 2 * output_dim, kernel_size, dilation=dilate, bias=False)
        self.padding_mode = padding_mode

        # learnable padding
        if padding_mode == "learnable":
            self.padding_vector = nn.Parameter(torch.empty(output_dim, 1))
            nn.init.kaiming_normal_(self.padding_vector)  # parameter initialization

        # skip connection to the end of **all the stacked dilated layers**
        # an additional parameterized shortcut
        # learnable regardless of dimensional expansion or contraction
        self.skip_connection = nn.Conv1d(output_dim, skip_dim, 1, 1, 0)

    def forward(self, x, y=None, automatic_padding=True):

        identity = x.clone()
        if self.residual_connection is not None:
            identity = self.residual_connection(identity)

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
                    padded = torch.cat([
                        self.padding_vector[None, :, :].repeat((x.size(0), 1, self.left_padding)),
                        x], dim=-1)
                elif isinstance(automatic_padding, int):
                    padded = torch.cat([
                        self.padding_vector[None, :, :].repeat((x.size(0), 1, automatic_padding)),
                        x], dim=-1)
                else:
                    raise ValueError("Unexpected type of automatic padding!")

        else:
            padded = x

        content, output = self.conv1(padded).chunk(2, dim=1)
        if self.condition != "none":
            content_cond, output_cond = self.conv3(y).chunk(2, dim=1)

            if self.condition == "global":
                # broadcast over time
                content_cond = content_cond.unsqueeze(2)
                output_cond = output_cond.unsqueeze(2)
            content = content + content_cond
            output = output + output_cond

        content = torch.tanh(content)  # tanh activation
        output = torch.sigmoid(output)  # sigmoid activation
        hidden_repr = content * output

        residual = self.conv2(hidden_repr)
        skip = self.skip_connection(hidden_repr)

        out = residual + identity[:, :, -residual.size(2):]  # residual connection
        return out, skip[:, :, -out.size(2):]


class WaveNet(nn.Module):
    def __init__(
            self,
            input_dim=256,
            hidden_dim=16,
            embed_dim=None,  # for conditioning only
            skip_dim=32,
            output_dim=256,
            layers_per_block=10,
            num_blocks=2,
            kernel_size=2,
            condition="none",
            quantization=True,
            padding_mode="constant",
            sampler=NucleusSampler()
    ):
        super(WaveNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.skip_dim = skip_dim
        self.output_dim = output_dim
        self.layers_per_block = layers_per_block
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.condition = condition
        self.padding_mode = padding_mode

        if quantization:
            # take input values (integer) as coding indices
            self.feature = nn.Embedding(input_dim+1, hidden_dim, padding_idx=0)
        else:
            self.feature = nn.Conv1d(input_dim, hidden_dim, 1, 1, 0)
        self.quantization = quantization

        for i in range(num_blocks):
            self.add_module(f"block_{i}", self._make_block())

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_dim, output_dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, 1, 1, 0)
        )

        if condition != "none":
            self.use_label = None  # indicator attribute

        self.sampler = sampler

    def _make_block(self):

        class Block(nn.Module):

            def __init__(
                    self,
                    num_layers,
                    kernel_size,
                    hidden_dim,
                    embed_dim,
                    skip_dim,
                    condition
            ):
                super(Block, self).__init__()
                """
                take kernel size 2 as example, 
                the ultimate receptive region of 10 stacked layers is 1024 pixel wide
                """
                for i in range(num_layers):
                    self.add_module(
                        f"dilated_causal_layer_{i}",
                        DilatedCausalConv1d(
                            kernel_size, kernel_size ** i, input_dim=hidden_dim, output_dim=hidden_dim,
                            skip_dim=skip_dim, embed_dim=embed_dim, condition=condition)
                    )
                self.num_layers = num_layers

            def forward(self, x, y=None):
                # culmulative skip connection
                skip_sum = 0
                for i in range(self.num_layers):
                    layer = getattr(self, f"dilated_causal_layer_{i}")
                    x, skip = layer(x, y)
                    skip_sum += skip
                return x, skip_sum

        return Block(self.layers_per_block, self.kernel_size,
                     self.hidden_dim, self.embed_dim, self.skip_dim, condition=self.condition)

    def generate(self, x, y=None, generate_length=0):
        layers = []
        n_samples, input_length = x.shape
        full_length = input_length + generate_length
        if y is not None:
            y = y.float()
        for block in [getattr(self, f"block_{i}") for i in range(self.num_blocks)]:
            layers.extend([getattr(block, f"dilated_causal_layer_{j}") for j in range(self.layers_per_block)])
        out = torch.zeros(n_samples, full_length).to(x)
        hidden_repr = torch.zeros(n_samples, len(layers) + 1, self.hidden_dim, full_length).to(x.device)
        skip_sum = torch.zeros(n_samples, self.skip_dim, full_length).to(x.device)
        with torch.no_grad():
            out[:, :input_length] = x
            if self.quantization:
                hidden_repr[:, 0, :, :input_length] = self.feature(x + 1).swapdims(1, 2)
            else:
                hidden_repr[:, 0, :, :input_length] = self.feature(x.unsqueeze(1))
            for i, layer in enumerate(layers):
                hidden_repr[:, 1 + i, :, :input_length], skip = layer(hidden_repr[:, i, :, :input_length], y)
                skip_sum[:, :, :input_length] += skip
            if input_length < full_length:
                logits = self.classifier(skip_sum[:, :, [input_length - 1]]).squeeze(-1)
                out[:, input_length] = self.sampler.sample(logits=logits)
                if not self.quantization:
                    out[:, input_length] = out[:, input_length] / (self.output_dim // 2) - 1
                for i in range(input_length, full_length):
                    if self.quantization:
                        hidden_repr[:, 0, :, i] = self.feature(out[:, i] + 1)
                    else:
                        hidden_repr[:, 0, :, i] = self.feature(out[:, None, [i]]).squeeze(-1)
                    for j, layer in enumerate(layers):
                        hidden_repr[:, 1 + j, :, [i]], skip = layer(
                            hidden_repr[:, j, :, max(0, i - layer.left_padding):(i + 1)],
                            y, automatic_padding=max(0, layer.left_padding - i))
                        skip_sum[:, :, [i]] += skip
                    if not i == (full_length - 1):
                        logits = self.classifier(skip_sum[:, :, [i]]).squeeze(-1)
                        out[:, i + 1] = self.sampler.sample(logits=logits)
                        if not self.quantization:
                            out[:, i + 1] = out[:, i + 1] / (self.output_dim // 2) - 1
                if not self.quantization:
                    out = torch.round((out + 1) * (self.output_dim // 2)).int()
            else:
                out = self.classifier(skip_sum[:, :, :]).max(dim=1)[1]
        return out

    def forward(self, x, y=None):
        if self.quantization:
            # reserve index 0 for padding token
            x = x + 1
        else:
            # expand additional representation dim
            x = x.unsqueeze(1)
        if y is not None:
            y = y.float()
        feat = self.feature(x)
        if self.quantization:
            feat = feat.swapdims(1, 2)
        skip_sum = 0
        for i in range(self.num_blocks):
            block = getattr(self, f"block_{i}")
            if i == 0:
                hidden_repr, skip = block(feat, y)
            else:
                hidden_repr, skip = block(hidden_repr, y)
            skip_sum += skip
        # feed the cumulative skip connection to the classifier
        out = self.classifier(skip_sum)
        return out
