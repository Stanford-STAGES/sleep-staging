from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.vae_components import (
    slasnista_encoder,
    slasnista_decoder,
)


class Encoder(nn.Module):
    def __init__(
        self, kernel_width: int = 3, n_filters: int = 128, n_layers: int = 2, **kwargs,
    ):
        super().__init__()
        self.kw = kernel_width
        self.nf = n_filters
        self.nl = n_layers

    def forward(self, x):
        raise NotImplementedError


class Decoder(nn.Module):
    def __init__(
        self, kernel_width: int = 3, latent_dim: int = 256, n_filters: int = 128, n_layers: int = 2, **kwargs,
    ):
        super().__init__()
        self.kw = kernel_width
        self.ld = latent_dim
        self.nf = n_filters
        self.nl = n_layers

    def forward(self, x):
        raise NotImplementedError


class VAE(pl.LightningModule):
    def __init__(
        self,
        kernel_width: int = 3,
        enc_type: str = "slasnista",
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-5,
        n_filters: int = 128,
        n_layers: int = 2,
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(
            self.hparams.batch_size, self.hparams.n_channels, self.hparams.sequence_length
        )

        valid_encoders = {"slasnista": {"enc": slasnista_encoder, "dec": slasnista_decoder}}

        if enc_type not in valid_encoders:
            self.encoder = Encoder(self.hparams)
            self.decoder = Decoder(self.hparams)
        else:
            self.encoder = valid_encoders[enc_type]["enc"](self.hparams)
            self.decoder = valid_encoders[enc_type]["dec"](self.hparams)

        self.fc_mu = nn.Linear(self.hparams.n_filters, self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.n_filters, self.hparams.latent_dim)

    def forward(self, x):
        raise NotImplementedError

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qz - log_pz).mean()
        kl *= self.hparams.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"eval_{k}": v for k, v in logs.items()})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--kernel_width", type=int, default=3)
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--n_filters", type=int, default=128)
        parser.add_argument("--n_layers", type=int, default=2)

        return parser


if __name__ == "__main__":

    pl.seed_everything(1337)

    parser = ArgumentParser()
    parser = VAE.add_model_specific_args(parser)
    args = parser.parse_args()
    args.batch_size = 16
    args.n_channels = 5
    args.sequence_length = 5 * 60 * 128

    model = VAE(**vars(args))
    print(model)

    x_shape = (args.batch_size, args.n_channels, args.sequence_length)
    x = torch.rand(x_shape)
    z = model()
