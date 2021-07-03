import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dst
from functools import reduce


class DecoderLayer(nn.Module):
    """
    Deep AutoRegressive Decoder Layer
    """
    def __init__(self, in_features, out_features, autoregress=True, activation=None):
        super(DecoderLayer, self).__init__()
        if in_features == 0:
            if autoregress is False:
                raise ValueError("Cannot create non-autoregressive layer for zero in_features!")
            self.decoder = nn.Conv1d(out_features, out_features, out_features, groups=out_features)
        else:
            # group 1d convolution
            # utilize the gralatentic processing unit for maximum speed gain
            self.decoder = nn.Conv1d(
                out_features, out_features,
                in_features + out_features - 1,  # kernel size
                groups=out_features
            ) if autoregress else nn.Linear(in_features, out_features)

        self.in_features = in_features
        self.out_features = out_features
        if autoregress:
            self.mask = nn.Parameter(torch.tril(
                    torch.ones(out_features, max(in_features - 1, 0) + out_features),
                    diagonal=max(in_features - 1, 0)
            ))  # mask for autoregression
            self.mask.requires_grad_(False)  # disable gradient
        self.autoregress = autoregress
        self.sampler = dst.bernoulli.Bernoulli
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        # output activation
        # for non-autoregressive layer only
        if activation is None:
            self.activation = lambda x: x  # identity function
        else:
            self.activation = activation

    def sample(self, n_samples, x=None, return_prob=False):
        assert self.autoregress, "Applied to autoregressive layer only!"
        if x is None:
            x = torch.zeros(n_samples, 1).to(self.decoder.weight.device)
        out, logits = self.forward(x)
        if return_prob:  # whether to return log of sampling probability
            log_prob = -self.bce_loss(out, logits).mean(dim=1)
            return out, log_prob
        return out

    def log_prob(self, targets, x=None):
        if self.in_features == 0:
            x = torch.zeros(targets.size(0), 1).to(self.decoder.weight.device)
        if self.autoregress:
            logits = self.decoder(torch.mul(
                self.mask[None, :, :],
                torch.cat([x, targets[:, :-1]], dim=1).unsqueeze(1).repeat((1, self.out_features, 1))
            )).squeeze(2)
        else:
            logits = self.forward(x)
        return -self.bce_loss(logits, targets).mean(dim=1)

    def forward(self, x):
        if self.autoregress:
            out = torch.zeros(x.size(0), self.out_features).to(x.device)
            logits = torch.zeros_like(out)
            unit_decoders = zip(
                self.decoder.weight.chunk(self.out_features, 0),
                self.decoder.bias.chunk(self.out_features, 0)
            )
            for i, (weight, bias) in enumerate(unit_decoders):
                logits[:, i] = torch.sum(
                    weight[:, 0, :max(self.in_features, 1) + i] * torch.cat([x, out[:, :i]], dim=1), dim=1
                ) + bias
                out[:, i] = self.sampler(logits=logits[:, i]).sample()
            return out, logits
        else:
            out = self.activation(self.decoder(x))
            return out


class DARN(nn.Module):
    def __init__(
            self,
            input_shape=(1, 28, 28),
            deterministic_dim=128,
            stochastic_dim=16,
            input_autoregressivity=True
    ):
        super(DARN, self).__init__()
        self.input_shape = input_shape
        self.input_dim = reduce(lambda x,y: x*y, input_shape, 1)
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.input_autoregressivity = input_autoregressivity
        # input -> deterministic -> stochastic (bottom-up)
        # non-autoregressive
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.deterministic_dim),
            nn.BatchNorm1d(self.deterministic_dim),
            nn.ReLU(),
            nn.Linear(self.deterministic_dim, self.stochastic_dim)
        )
        # stochastic -> deterministic -> input (up-down)
        self.decoder = self._make_decoder()
        # prior
        self.prior = DecoderLayer(0, self.stochastic_dim)
        self.sampler = dst.bernoulli.Bernoulli
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def _make_decoder(self):

        class DARNDecoder(nn.Module):

            def __init__(
                    self,
                    input_dim,
                    deterministic_dim,
                    stochastic_dim,
                    input_autoregressivity
            ):
                super(DARNDecoder, self).__init__()
                # stochastic -> deterministic (non-autoregressive)
                self.layer1 = DecoderLayer(stochastic_dim, deterministic_dim, autoregress=False, activation=F.relu)
                # deterministic -> input (dependent on input_autoregressivity)
                self.layer2 = DecoderLayer(deterministic_dim, input_dim, autoregress=input_autoregressivity)

            def forward(self, latent, x=None):
                feature = self.layer1(latent)
                if x is None:
                    if self.layer2.autoregress:
                        _, logits = self.layer2(feature)
                    else:
                        logits = self.layer2(feature)
                    return logits
                else:
                    log_sampling = self.layer2.log_prob(x, feature)
                    return log_sampling

        return DARNDecoder(
            self.input_dim, self.deterministic_dim, self.stochastic_dim, self.input_autoregressivity)

    def generate(self, n_samples=1):
        latent = self.prior.sample(n_samples)
        input_logits = self.decoder(latent)
        return self.sampler(logits=input_logits).sample().reshape(-1, *self.input_shape).cpu()

    def forward(self, x):
        x_bin = self.sampler(probs=x.flatten(start_dim=1)).sample()  # binarization
        latent_logits = self.encoder(x_bin)
        latent = self.sampler(logits=latent_logits).sample()
        log_prior = self.prior.log_prob(latent)
        log_sampling = self.decoder(latent, x_bin)
        log_posterior = -self.bce_loss(latent_logits, latent).mean(dim=1)
        return (log_posterior - log_prior - log_sampling).mean()
