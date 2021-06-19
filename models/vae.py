import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(
            self,
            in_channels=1,
            input_size=(28, 28),
            latent_dim=64
    ):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.encoder_mean, self.encoder_logvar = self._get_encoder(), self._get_encoder()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.latent_dim, self.latent_dim // 2,
                tuple(map(lambda x: x // 4, self.input_size))
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.latent_dim // 2, self.latent_dim // 4, 4, 2, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.latent_dim // 4, self.in_channels, 6, 2, 2)
        )
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def _get_encoder(self):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, self.latent_dim // 4, 6, 2, 2),
            nn.BatchNorm2d(self.latent_dim // 4),
            nn.ReLU(),
            nn.Conv2d(self.latent_dim // 4, self.latent_dim // 2, 4, 2, 1),
            nn.BatchNorm2d(self.latent_dim // 2),
            nn.ReLU(),
            nn.Conv2d(self.latent_dim // 2, self.latent_dim, 3, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )

    def encode(self, x):
        mean, logvar = self.encoder_mean(x), self.encoder_logvar(x)
        latent = mean + torch.exp(logvar / 2) * torch.randn(x.size(0), self.latent_dim).to(x)
        # note that the integrals(expectations) have closed forms for Gaussian distribution
        # using closed-form expressions can improve the stability of the algorithm
        log_posterior = torch.sum(-1 / 2 * (1 + logvar), dim=1)
        log_prior = torch.sum(-1 / 2 * (mean ** 2 + torch.exp(logvar)), dim=1)

        return latent, log_posterior, log_prior

    def forward(self, x):
        latent, log_posterior, log_prior = self.encode(x)
        logits = self.decoder(latent.view(x.size(0), self.latent_dim, 1, 1))
        log_sampling = -self.bce_loss(logits, x).sum(dim=(1, 2, 3))  # reconstruction loss
        return (log_posterior - log_prior - log_sampling).mean()

    def generate(self, n_samples=1):
        latent = torch.randn(n_samples, self.latent_dim, 1, 1).to(next(self.parameters()).device)
        logits = self.decoder(latent)
        out = torch.sigmoid(logits).reshape(n_samples, self.in_channels, *self.input_size)
        return out.cpu()
