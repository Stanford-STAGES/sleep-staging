from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F


class SlasnistaEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # fmt: off
        self.eeg_eog_stream = nn.Sequential(OrderedDict([
            ('eeg_eog:zeropad_01', nn.ZeroPad2d((self.kernel // 2 - 1, self.kernel // 2, 0, 0))),
            ('eeg_eog:conv_01', nn.Conv2d(
                in_channels=1,
                out_channels=self.n_filters,
                kernel_size=(1, self.kernel)
            )),
            ('eeg_eog:relu_01', nn.ReLU()),
            ('eeg_eog:maxpool_01', nn.MaxPool2d(
                kernel_size=(1, self.maxpool),
                stride=(1, self.maxpool)
            )),
            ('eeg_eog:zeropad_02', nn.ZeroPad2d((self.kernel // 2 - 1, self.kernel // 2, 0, 0))),
            ('eeg_eog:conv_02', nn.Conv2d(
                in_channels=self.n_filters,
                out_channels=self.n_filters,
                kernel_size=(1, self.kernel)
            )),
            ('eeg_eog:relu_02', nn.ReLU()),
            ('eeg_eog:maxpool_02', nn.MaxPool2d(
                kernel_size=(1, self.maxpool),
                tride=(1, self.maxpool)
            )),
        ]))
        self.emg_stream = nn.Sequential(OrderedDict([
            ('emg:zeropad_01', nn.ZeroPad2d((self.kernel // 2 - 1, self.kernel // 2, 0, 0))),
            ('emg:conv_01', nn.Conv2d(
                in_channels=1,
                out_channels=self.n_filters,
                kernel_size=(1, self.kernel)
            )),
            ('emg:relu_01', nn.ReLU()),
            ('emg:maxpool_01', nn.MaxPool2d(
                kernel_size=(1, self.maxpool),
                stride=(1, self.maxpool)
            )),
            ('emg:zeropad_02', nn.ZeroPad2d((self.kernel // 2 - 1, self.kernel // 2, 0, 0))),
            ('emg:conv_02', nn.Conv2d(
                in_channels=self.n_filters,
                out_channels=self.n_filters,
                kernel_size=(1, self.kernel)
            )),
            ('emg:relu_02', nn.ReLU()),
            ('emg:maxpool_02', nn.MaxPool2d(
                kernel_size=(1, self.maxpool),
                stride=(1, self.maxpool)
            )),
        ]))
        # fmt: on

    def forward(x):
        pass
        eeg_eog, emg = torch.split(x.unsqueeze(2), [4, 1], dim=1)


class SlasnistaDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def forward(x):
        pass


def slasnista_encoder(hparams):
    return SlasnistaEncoder(hparams)


def slasnista_decoder(hparams):
    return SlasnistaDecoder(hparams)
