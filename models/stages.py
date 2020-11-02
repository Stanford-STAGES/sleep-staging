from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as ptl
from pytorch_lightning.metrics import Accuracy
#from pytorch_lightning.metrics import F1
from pytorch_lightning.metrics import Precision
from pytorch_lightning.metrics import Recall

from dataset import StagesData


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = [0, self.kernel_size[1] // 2, self.kernel_size[2] // 2]

        self.layers = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv3d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   stride=self.stride)),
                ('batch_norm', nn.BatchNorm3d(num_features=out_channels)),
                ('relu', nn.ReLU()),
            ])
        )

        # Initialize weights according to old code
        nn.init.normal_(self.layers[0].weight, 1e-3)
        nn.init.constant_(self.layers[0].bias, 0.0)

    def forward(self, x):
        return self.layers(x)


class LargeAutocorr(nn.Module):
    def __init__(self, model_name, modality, seg_size):
        super().__init__()
        self.model_name = model_name
        self.modality = modality
        self.seg_size = seg_size

        if modality == 'eeg':
            np.random.seed(int(self.model_name[-2:]) + 1)
            self.n_in = 2
            self.n_out = [64, 128, 256, 512]
            self.strides = [[3, 2], [2, 1]]
        elif modality == 'eog':
            np.random.seed(int(self.model_name[-2:]) + 2)
            self.n_in = 3
            self.n_out = [64, 128, 256, 512]
            self.strides = [[4, 2], [2, 1]]
        elif modality == 'emg':
            np.random.seed(int(self.model_name[-2:]) + 3)
            self.n_in = 1
            self.n_out = [16, 32, 64, 128]
            self.strides = [[2, 2], [2, 1]]
        else:
            raise ValueError(f'Please specify modality, got {modality}.')

        self.layers = nn.Sequential(OrderedDict([
            (f'conv-01-{modality}', Conv2DBlock(self.n_in, self.n_out[0], [1, 7, 7], [1] + self.strides[0])),
            (f'maxpool-01-{modality}', nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 0, 1])),
            (f'conv-02-{modality}', Conv2DBlock(self.n_out[0], self.n_out[1], [1, 5, 5], [1] + self.strides[1])),
            (f'conv-03-{modality}', Conv2DBlock(self.n_out[1], self.n_out[1], [1, 3, 3], [1, 1, 1])),
            (f'maxpool-02-{modality}', nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 1, 1])),
            (f'conv-04-{modality}', Conv2DBlock(self.n_out[1], self.n_out[2], [1, 3, 3], [1, 1, 1])),
            (f'conv-05-{modality}', Conv2DBlock(self.n_out[2], self.n_out[2], [1, 3, 3], [1, 1, 1])),
            (f'maxpool-03-{modality}', nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 1, 0])),
            (f'conv-06-{modality}', Conv2DBlock(self.n_out[2], self.n_out[3], [1, 3, 3], [1, 1, 1])),
            (f'conv-07-{modality}', Conv2DBlock(self.n_out[3], self.n_out[3], [1, 3, 3], [1, 1, 1])),
        ]))

    def forward(self, x):
        batch_size, n_segs, seg_size, n_features = x.shape
        z = x.reshape(batch_size, n_segs, seg_size, self.n_in, -1).permute(0, 3, 1, 4, 2)
        for layer in self.layers:
            z = layer(z)
        z = torch.mean(z, dim=3)
        z = torch.mean(z, dim=3)
        return z


class RandomAutocorr(torch.nn.Module):
    def __init__(self, model_name, modality, seg_size):
        super().__init__()
        self.model_name = model_name
        self.modality = modality
        self.seg_size = seg_size

        if modality == 'eeg':
            np.random.seed(int(self.model_name[-2:]) + 1)
            self.n_in = 2
            self.n_out = [np.random.randint(32, 96),
                          np.random.randint(64, 192),
                          np.random.randint(128, 384)]
            self.strides = [[3, 2], [2, 1]]
        elif modality == 'eog':
            np.random.seed(int(self.model_name[-2:]) + 2)
            self.n_in = 3
            self.n_out = [np.random.randint(32, 96),
                          np.random.randint(64, 192),
                          np.random.randint(128, 384)]
            self.strides = [[4, 2], [2, 1]]
        elif modality == 'emg':
            np.random.seed(int(self.model_name[-2:]) + 3)
            self.n_in = 1
            self.n_out = [np.random.randint(8, 24),
                          np.random.randint(16, 48),
                          np.random.randint(32, 96)]
            self.strides = [[2, 2], [2, 1]]
        else:
            raise ValueError(f'Please specify modality, got {modality}.')

        self.layers = nn.Sequential(OrderedDict([
            (f'conv-01-{modality}', Conv2DBlock(self.n_in, self.n_out[0], [1, 7, 7], [1] + self.strides[0])),
            (f'maxpool-01-{modality}', nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])),
            (f'conv-02-{modality}', Conv2DBlock(self.n_out[0], self.n_out[1], [1, 5, 5], [1] + self.strides[1])),
            (f'conv-03-{modality}', Conv2DBlock(self.n_out[1], self.n_out[1], [1, 3, 3], [1, 1, 1])),
            (f'maxpool-02-{modality}', nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 1, 1])),
            (f'conv-04-{modality}', Conv2DBlock(self.n_out[1], self.n_out[2], [1, 3, 3], [1, 1, 1])),
            (f'conv-05-{modality}', Conv2DBlock(self.n_out[2], self.n_out[2], [1, 3, 3], [1, 1, 1])),
        ]))

    def forward(self, x):
        batch_size, n_segs, seg_size, n_features = x.shape
        z = x.reshape(batch_size, n_segs, seg_size, self.n_in, -1).permute(0, 3, 1, 4, 2)
        for layer in self.layers:
            z = layer(z)
        z = torch.mean(z, dim=3)
        z = torch.mean(z, dim=3)
        return z


class StagesModel(ptl.LightningModule):

    # def __init__(self, batch_size=5, data_dir=None, dropout=None, eval_ratio=0.1, learning_rate=0.005, model_name=None,
    #              momentum=0.9, n_classes=None, n_hidden_units=None, n_jobs=None, n_workers=None, seg_size=None, **kwargs):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.hparams = kwargs
        # self.batch_size = batch_size
        # self.data_dir = data_dir
        # self.dropout = dropout
        # self.eval_ratio = eval_ratio
        # self.learning_rate = learning_rate
        # self.model_name = model_name
        # self.momentum = momentum
        # self.n_classes = n_classes
        # self.n_hidden_units = n_hidden_units
        # self.n_jobs = n_jobs
        # self.n_workers = n_workers
        # self.seg_size = seg_size
        # self.weight_decay = weight_decay
        # self.metrics = {
        #     'accuracy': Accuracy(reduction='none'),
        #     'f1': F1(reduction='none'),
        #     'precision': Precision(reduction='none'),
        #     'recall': Recall(reduction='none'),
        # }
        # self.metrics = {
        #     'accuracy': Accuracy(num_classes=self.n_classes, reduction='none'),
        #     'f1': F1(num_classes=self.n_classes, reduction='none'),
        #     'precision': Precision(num_classes=self.n_classes, reduction='none'),
        #     'recall': Recall(num_classes=self.n_classes, reduction='none'),
        # }

        self.gamma = np.exp(-1/750)
        self.lstm = self.model_name.split('_')[3] == 'lstm'
        self.steps_per_file = 1 # 300 // self.batch_size

        if self.model_name.split('_')[1] == 'rh':
            self.hidden_eeg = RandomAutocorr(self.model_name, 'eeg', self.seg_size)
            self.hidden_eog = RandomAutocorr(self.model_name, 'eog', self.seg_size)
            self.hidden_emg = RandomAutocorr(self.model_name, 'emg', self.seg_size)
        elif self.model_name.split('_')[1] == 'lh':
            self.hidden_eeg = LargeAutocorr(self.model_name, 'eeg', self.seg_size)
            self.hidden_eog = LargeAutocorr(self.model_name, 'eog', self.seg_size)
            self.hidden_emg = LargeAutocorr(self.model_name, 'emg', self.seg_size)
        self.hparams.n_out_eeg = self.hidden_eeg.n_out
        self.hparams.n_out_eog = self.hidden_eog.n_out
        self.hparams.n_out_emg = self.hidden_emg.n_out

        self.n_hidden_features = (self.hidden_eeg.layers[-1].layers[0].weight.shape[1] +
                                  self.hidden_eog.layers[-1].layers[0].weight.shape[1] +
                                  self.hidden_emg.layers[-1].layers[0].weight.shape[1])

        if self.lstm:
            self.hidden_hidden = nn.ModuleList([
                nn.Dropout(p=1 - self.dropout),
                torch.nn.LSTM(self.n_hidden_features, self.n_hidden_units, bias=True, batch_first=True),
                nn.Dropout(p=1 - self.dropout)
            ])

        # Classification layer
        self.output_layer = nn.Linear(self.n_hidden_units, self.n_classes, bias=True)
        nn.init.normal_(self.output_layer.weight, std=0.04)
        nn.init.constant_(self.output_layer.bias, 0.001)

    def forward(self, x):
        if self.lstm:
            self.hidden_hidden[1].flatten_parameters()

        x_eeg, x_eog, x_emg = torch.split(x, [400, 1200, 40], dim=-1)

        # Feature extraction
        z_eeg = self.hidden_eeg(x_eeg)
        z_eog = self.hidden_eog(x_eog)
        z_emg = self.hidden_emg(x_emg)
        z = torch.cat([z_eeg, z_eog, z_emg], dim=1).permute([0, 2, 1])
        del z_eeg, z_eog, z_emg, x_eeg, x_eog, x_emg, x

        # Feature processing
        if self.lstm:
            z = self.hidden_hidden[0](z) # Dropout on input
            z, _ = self.hidden_hidden[1](z) # LSTM
            z = self.hidden_hidden[2](z) # Dropout on output
        else:
            raise NotImplementedError

        # Output layer
        z = z.reshape([-1, self.n_hidden_units])
        logits = self.output_layer(z)

        return logits

    def compute_loss(self, y, y_hat):
        y = y.mean(dim=2).reshape(-1, self.n_classes)
        loss = F.cross_entropy(torch.clamp(y_hat, min=-1e7, max=1e7), y.argmax(dim=1))
        if torch.isnan(loss).any():
            print('Bug!')
        return loss

    def compute_metrics(self, y, y_hat):
        y = y.mean(dim=2).reshape(-1, self.n_classes)
        softmax = F.softmax(y_hat, dim=-1)
        metrics = {'accuracy': (softmax.argmax(dim=-1) == y.argmax(dim=-1)).to(torch.float32).mean()}
        # per_class_metrics = {metric: metric_fn(softmax.argmax(dim=1), y.argmax(dim=1)) for metric, metric_fn in self.metrics.items()}
        # metrics = {'_'.join([k, stage]): per_class_metrics[k][idx] for k in ['f1', 'precision', 'recall'] for idx, stage in enumerate(['w', 'n1', 'n2', 'n3', 'rem'])}
        # metrics.update({k: v.mean() for k, v in per_class_metrics.items()})
        # metrics['accuracy'] = per_class_metrics['accuracy'].mean()
        # metrics['baseline'] = y.mean(dim=0).detach()
        return metrics

    def training_step(self, batch, batch_index):

        X, y, w = batch
        # del batch
        if torch.isnan(X).any():
            X[torch.isnan(X)] = torch.tensor(2 ** -23, device=self.device) # Hack to fix zero division from encoding
        y_hat = self.forward(X)
        if torch.isnan(y_hat).any():
            print('Bug!')
        loss = self.compute_loss(y, y_hat)
        metrics = {'_'.join(['train', k]): v for k, v in self.compute_metrics(y, y_hat).items()}

        # logs = {'train_loss': loss, 'train_acc': acc, 'train_baseline': baseline}

        return {'loss': loss, 'log': {**dict(train_loss=loss), **metrics}}

    def validation_step(self, batch, batch_index):

        X, y, w = batch
        del batch
        y_hat = self.forward(X)
        loss = self.compute_loss(y, y_hat)
        metrics = {'_'.join(['eval', k]): v for k, v in self.compute_metrics(y, y_hat).items()}

        return {**dict(eval_loss=loss), **metrics}
        # return {'eval_loss': loss}.update({'_'.join(['eval', k]): v for k, v in metrics.items()})
        # return {'val_loss': loss, 'val_acc': acc, 'val_baseline': baseline}

    def validation_epoch_end(self, outputs):

        return {'val_loss': torch.stack([x['eval_loss'] for x in outputs]).mean(),
                'log': {k: torch.stack([x[k] for x in outputs]).mean() for k in outputs[0].keys()}}

    def configure_optimizers(self):

        optimizer = torch.optim.SGD([p[1] for p in self.named_parameters(
        ) if not 'bias' in p[0] and not 'batch_norm' in p[0]], lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, self.gamma),
                     'interval': 'epoch',
                     'frequency': 1, # self.steps_per_file,
                     'name': 'lr_schedule'}
        # scheduler = None
        # return optimizer
        return [optimizer], [scheduler]

    # def collate_fn(self, batch):

    #     X, y, w = [torch.FloatTensor(b) for b in batch]

    #     return X.squeeze(), y.squeeze(), w.squeeze()

    def train_dataloader(self):
        """Return training dataloader. Batching is handled in the __getitem__ of the dataset."""
        return torch.utils.data.DataLoader(self.train_data, batch_size=None, shuffle=False, num_workers=self.n_workers, pin_memory=True)
        # return torch.utils.data.DataLoader(self.train_data, batch_size=None, collate_fn=self.collate_fn, shuffle=False, num_workers=self.n_workers, pin_memory=True)

    def val_dataloader(self):
        """Return validation dataloader. Batching is handled in the __getitem__ of the dataset."""
        # return torch.utils.data.DataLoader(self.eval_data, batch_size=None, collate_fn=self.collate_fn, shuffle=False, num_workers=self.n_workers, pin_memory=True)
        return torch.utils.data.DataLoader(self.eval_data, batch_size=None, shuffle=False, num_workers=self.n_workers, pin_memory=True)

    # def prepare_data(self):
        # print('Hej')
        # self.dataset_params = dict(
        #     data_dir=self.data_dir, n_classes=self.n_classes, n_jobs=self.n_jobs, seg_size=self.seg_size)
        # self.dataset = StagesData(**self.dataset_params)
        # self.split_data()
        # self.train_data, self.eval_data = self.dataset.split_data(0.1)
        # print(self.train_data)
        # print(self.eval_data)

    def setup(self, stage):
        print('Setup')
        self.dataset_params = dict(
            data_dir=self.data_dir, n_classes=self.n_classes, n_jobs=self.n_jobs, seg_size=self.seg_size)
        self.dataset = StagesData(**self.dataset_params)
        self.train_data, self.eval_data = self.dataset.split_data(self.eval_ratio)

    def split_data(self):

        n_total = len(self.dataset)
        n_eval = int(np.ceil(self.eval_ratio * n_total))
        n_train = n_total - n_eval

        self.train_data, self.eval_data = torch.utils.data.random_split(self.dataset, [
                                                                        n_train, n_eval])
        print('Dataset length: ', len(self.dataset))
        print('Train dataset length: ', len(self.train_data))
        print('Eval dataset length: ', len(self.eval_data))

    def on_post_performance_check(self):
        self.train_data, self.eval_data = self.dataset.split_data(self.eval_ratio)
        print(self.train_data)
        print(self.eval_data)

    @staticmethod
    def add_model_specific_args(parent_parser):

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', default=5, type=int)
        parser.add_argument('--dropout', default=0.5, type=float)
        parser.add_argument('--model_name', default='ac_rh_ls_lstm_01', type=str)
        parser.add_argument('--n_hidden_units', default=128, type=int)
        parser.add_argument('--n_classes', default=5, type=int)
        parser.add_argument('--seg_size', default=60, type=int)

        # OPTIMIZER specific
        parser.add_argument('--learning_rate', default=0.005, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--weight_decay', default=1e-5, type=float)

        # Data specific
        parser.add_argument('--data_dir', default='data/train_data', type=str)
        parser.add_argument('--eval_ratio', default=0.1, type=float)
        parser.add_argument('--n_jobs', default=-1, type=int)
        parser.add_argument('--n_workers', default=10, type=int)

        return parser


if __name__ == '__main__':

    model_parameters = dict(model_name='ac_lh_ls_lstm_01', n_hidden_units=128,
                            seg_size=60, dropout=0.5, n_classes=5, n_jobs=-1, data_dir='data/train_data')
    model = StagesModel(**model_parameters)
    model.configure_optimizers()
    model.prepare_data()
    model_summary = ptl.core.memory.ModelSummary(model, 'top')
    print(model_summary)

    seg_size = 60
    n_segs = 20
    batch_size = 5
    n_features = 400 + 1200 + 40
    x = torch.rand([batch_size, n_segs, seg_size, n_features])
    z = model(x)
    print('z.shape:', z.shape)
    pass
