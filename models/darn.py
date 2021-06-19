import torch
import torch.nn as nn
import torch.distributions as dst
from functools import reduce


class DARN(nn.Module):
    def __init__(
            self,
            in_channels=1,
            input_size=(28, 28),
            deterministic_dim=512,
            stochastic_dim=32,
            memory_efficient=False,
            skip_connection=False
    ):
        super(DARN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, deterministic_dim // 2, 6, 2, 2),
            nn.BatchNorm2d(deterministic_dim // 2),
            nn.ReLU(),
            nn.Conv2d(deterministic_dim // 2, deterministic_dim, 4, 2, 1),
            nn.BatchNorm2d(deterministic_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )  # encoding deterministic hidden layer (bottom-up)
        self.inv_feature = nn.Sequential(
            nn.Linear(stochastic_dim, deterministic_dim),
            nn.BatchNorm1d(deterministic_dim),
            nn.ReLU()
        )

        self.in_channels = in_channels
        self.input_size = input_size

        self.input_dim = in_channels * reduce(lambda x, y: x * y, input_size, 1)
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim

        self.encoder = nn.Linear(deterministic_dim, stochastic_dim)

        self.memory_efficient = memory_efficient  # Set false to optimize for training speed
        self.skip_connection = skip_connection

        self._add_autoregressor()

        self.log_sigmoid = nn.LogSigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        self._sampler = dst.bernoulli.Bernoulli

    def _add_autoregressor(self):
        if self.memory_efficient:
            for i in range(self.input_dim):
                self.add_module(f"px_{i}_logit", nn.Linear(self.deterministic_dim + i, 1))  # decoder
        else:
            # group 1d convolution
            # utilize the graphic processing unit for maximum speed gain
            self.decoder = nn.Conv1d(
                self.input_dim, self.input_dim,
                self.deterministic_dim + self.input_dim - 1,  # kernel size
                groups=self.input_dim
            )
        for i in range(self.stochastic_dim):
            if i == 0:
                setattr(self, "ph_0_logit", nn.Parameter(torch.Tensor([0.0])))
            else:
                self.add_module(f"ph_{i}_logit", nn.Linear(i, 1))  # latent prior

    def _prior_logprob(self, latent):
        log_prior = torch.zeros(latent.size(0), ).to(latent)
        for i in range(self.stochastic_dim):
            if i == 0:
                log_prior += self.log_sigmoid((2 * latent[:, 0] - 1) * getattr(self, "ph_0_logit"))
            else:
                log_prior += self.log_sigmoid(
                    (2 * latent[:, i] - 1) * getattr(self, f"ph_{i}_logit")(latent[:, :i]).ravel()
                )
        return log_prior

    def _sampling_logprob(self, x, latent):
        feat = self.inv_feature(latent)
        log_sampling = torch.zeros(latent.size(0), ).to(feat)
        if self.memory_efficient:
            for i in range(self.input_dim):
                logits = getattr(self, f"px_{i}_logit")(torch.cat([feat, x[:, :i]], dim=1))
                log_sampling += -self.bce_loss(logits, x[:, [i]]).ravel()
        else:
            mask = torch.tril(
                torch.ones(self.input_dim, self.deterministic_dim + self.input_dim - 1),
                diagonal=self.deterministic_dim-1
            ).to(x)
            logits = self.decoder(
                mask[None, :, :] * torch.cat([feat, x[:, :-1]], dim=1).unsqueeze(1).repeat(1, self.input_dim, 1)
            ).squeeze(2)
            if self.skip_connection:
                logits[:, 1:] += logits[:, :-1]
            log_sampling = -self.bce_loss(logits, x).sum(dim=1)
        return log_sampling

    def _encode(self, x):
        feat = self.feature(x)
        logits = self.encoder(feat)
        latent = self._sampler(logits=logits.detach()).sample()
        log_posterior = torch.sum(self.log_sigmoid(
            (2 * latent - 1) * logits
        ), dim=1)
        return latent, log_posterior

    def _sample_from_prior(self, n_samples=1):
        latent = torch.zeros(n_samples, self.stochastic_dim).to(getattr(self, "ph_0_logit"))
        with torch.no_grad():
            for i in range(self.stochastic_dim):
                if i == 0:
                    latent[:, i] = self._sampler(logits=getattr(self, "ph_0_logit")).sample((n_samples, )).ravel()
                else:
                    latent[:, i] = self._sampler(logits=getattr(self, f"ph_{i}_logit")(latent[:, :i])).sample().ravel()
            return latent

    def generate(self, n_samples=1):
        latent = self._sample_from_prior(n_samples)
        out = torch.zeros(n_samples, self.input_dim).to(getattr(self, "ph_0_logit"))
        with torch.no_grad():
            feat = self.inv_feature(latent)
            if self.memory_efficient:
                for i in range(self.input_dim):
                    if self.skip_connection and i > 0:
                        identity = out[:, i-1]
                    else:
                        identity = 0
                    out[:, i] = self._sampler(
                        logits=getattr(self, f"px_{i}_logit")(torch.cat([feat, out[:, :i]], dim=1)) + identity
                    ).sample().ravel()
            else:
                unit_decoders = zip(
                    self.decoder.weight.chunk(self.input_dim, 0),
                    self.decoder.bias.chunk(self.input_dim, 0)
                )
                for i, (weight, bias) in enumerate(unit_decoders):
                    if self.skip_connection and i > 0:
                        identity = out[:, i-1]
                    else:
                        identity = 0
                    out[:, i] = self._sampler(
                        logits=torch.sum(
                            weight[:, 0, :self.deterministic_dim+i] * torch.cat([feat, out[:, :i]], dim=1), dim=1
                        ) + bias + identity
                    ).sample().ravel()
        return out.reshape(n_samples, self.in_channels, *self.input_size).cpu()

    def forward(self, x):
        latent, log_posterior = self._encode(x)  # encoder
        log_prior = self._prior_logprob(latent)
        log_sampling = self._sampling_logprob(x.flatten(start_dim=1), latent)  # decoder
        loss = torch.mean(log_posterior - log_prior - log_sampling)
        return loss
