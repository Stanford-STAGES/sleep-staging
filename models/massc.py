import os
import pickle
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as ptl
from joblib import delayed
from pytorch_lightning.metrics import Accuracy
from sklearn import metrics
from tqdm import tqdm

# from pytorch_lightning.metrics import F1
# from pytorch_lightning.metrics import Precision
# from pytorch_lightning.metrics import Recall

# import datasets
# import utils
try:
    from utils import ParallelExecutor
except:
    from utils.parallel_bar import ParallelExecutor

class MasscModel(ptl.LightningModule):

# fmt: off
class MasscModel(ptl.LightningModule):
    def __init__(
        self, hparams, *args, **kwargs
        # self, filter_base=None, kernel_size=None, max_pooling=None, n_blocks=None, n_channels=None,
        # n_classes=None, n_rnn_layers=None, n_rnn_units=None, rnn_bidirectional=None, rnn_dropout=None,
        # optimizer=None, learning_rate=None, momentum=None, weight_decay=None, loss_weight=None, lr_scheduler=None,
        # base_lr=None, lr_reduce_factor=None, lr_reduce_patience=None, max_lr=None, step_size_up=None,
        # data_dir=None, eval_ratio=None, n_jobs=None, n_records=None, scaling=None, adjustment=None,
        # batch_size=None, n_workers=None, **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters({k: v for k, v in hparams.items() if not callable(v)})
        self.train_acc = Accuracy()
        # self.metrics = {
        #     "accuracy": Accuracy(),
        #     "f1": F1(),
        #     "precision": Precision(),
        #     "recall": Recall(),
        #     "accuracy": Accuracy(reduce_op="mean"),
        #     "f1": F1(reduce_op="mean"),
        #     "precision": Precision(reduce_op="mean"),
        #     "recall": Recall(reduce_op="mean"),
        #     "accuracy": Accuracy(num_classes=self.hparams.n_classes, reduce_op="mean"),
        #     "f1": F1(num_classes=self.hparams.n_classes, reduce_op="mean"),
        #     "precision": Precision(num_classes=self.hparams.n_classes, reduce_op="mean"),
        #     "recall": Recall(num_classes=self.hparams.n_classes, reduce_op="mean"),
        # }
        self.example_input_array = torch.zeros(self.hparams.batch_size, self.hparams.n_channels, 5 * 60 * 128)

        # Create mixing block
        if self.hparams.n_channels != 1:
            self.mixing_block = nn.Sequential(OrderedDict([
                ('mix_conv', nn.Conv1d(self.hparams.n_channels, self.hparams.n_channels, 1)),
                ('mix_batchnorm', nn.BatchNorm1d(self.hparams.n_channels)),
                ('mix_relu', nn.ReLU())
            ]))

        # Create shortcuts
        self.shortcuts = nn.ModuleList([
            nn.Sequential(OrderedDict([
                (f'shortcut_conv_{k}', nn.Conv1d(
                    in_channels=self.hparams.n_channels if k == 0 else 4 * self.hparams.filter_base * (2 ** (k - 1)),
                    out_channels=4 * self.hparams.filter_base * (2 ** k),
                    kernel_size=1
                )),
            ])) for k in range(self.hparams.n_blocks)
        ])

        # Create basic block structure
        self.blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                (f'conv_{k}_1', nn.Conv1d(
                    in_channels=self.hparams.n_channels if k == 0 else 4 * self.hparams.filter_base * (2 ** (k - 1)),
                    out_channels=self.hparams.filter_base * (2 ** k),
                    kernel_size=1)),
                (f'batchnorm_{k}_1', nn.BatchNorm1d(self.hparams.filter_base * (2 ** k))),
                (f'relu_{k}_1', nn.ReLU()),
                (f'conv_{k}_2', nn.Conv1d(
                    in_channels=self.hparams.filter_base * (2 ** k),
                    out_channels=self.hparams.filter_base * (2 ** k),
                    kernel_size=self.hparams.kernel_size,
                    padding=self.hparams.kernel_size // 2)),
                (f'batchnorm_{k}_2', nn.BatchNorm1d(self.hparams.filter_base * (2 ** k))),
                (f'relu_{k}_2', nn.ReLU()),
                (f'conv_{k}_3', nn.Conv1d(
                    in_channels=self.hparams.filter_base * (2 ** k),
                    out_channels=4 * self.hparams.filter_base * (2 ** k),
                    kernel_size=1)),
                (f'batchnorm_{k}', nn.BatchNorm1d(4 * self.hparams.filter_base * (2 ** k)))
            ])) for k in range(self.hparams.n_blocks)
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=self.hparams.max_pooling)
        self.relu = nn.ReLU()

        # Create temporal processing block
        if self.hparams.n_rnn_units > 0:
        self.temporal_block = nn.GRU(
                input_size=4 * self.hparams.filter_base * (2 ** (self.hparams.n_blocks - 1)),
                hidden_size=self.hparams.n_rnn_units,
                num_layers=self.hparams.n_rnn_layers,
            batch_first=True,
                dropout=self.hparams.rnn_dropout,
                bidirectional=self.hparams.rnn_bidirectional
        )
            classification_in_channels = (1 + self.hparams.rnn_bidirectional) * self.hparams.n_rnn_units
        else:
            self.temporal_block = None
            classification_in_channels = 4 * self.hparams.filter_base * (2 ** (self.hparams.n_blocks - 1))

        # Create classification block
        self.classification = nn.Conv1d(
            in_channels=classification_in_channels,
            out_channels=self.hparams.n_classes,
            kernel_size=1
        )
        # Init bias term to 1 / n_classes
        nn.init.constant_(self.classification.bias, 1 / self.hparams.n_classes)

        # Define loss function
        # self.register_buffer('loss_weights', torch.Tensor(self.hparams.cb_weights))
        # self.loss = nn.CrossEntropyLoss(weight=self.loss_weights)
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(self.hparams.cb_weights))

        # Dataset params
        self.dataset_params = dict(
            data_dir=self.hparams.data_dir,
            n_jobs=self.hparams.n_jobs,
            n_records=self.hparams.n_records,
            scaling=self.hparams.scaling,
            adjustment=self.hparams.adjustment,
        )

        # fmt: on

    def forward(self, x):
        if self.temporal_block:
            self.temporal_block.flatten_parameters()

        # Mixing block
        z = self.mixing_block(x)

        # Feature extraction block
        for block, shortcut in zip(self.blocks, self.shortcuts):
            y = shortcut(z)
            z = block(z)
            z += y
            z = self.relu(z)
            z = self.maxpool(z)

        # Temporal processing block
        if self.temporal_block:
            z = self.temporal_block(z.squeeze(2).transpose(1, 2))
            z = z[0].transpose(1, 2)
        else:
            z = z.squeeze(2)

        # Classification block
        z = self.classification(z)

        return z

    def compute_loss(self, y, y_hat):
        loss = self.loss(y_hat, y.argmax(dim=1))
        # if self.hparams.loss_weight:
        #     loss = F.cross_entropy(y_hat, y.argmax(dim=1), weight=torch.Tensor(self.hparams.loss_weight).type_as(y_hat))
        # else:
        #     loss = F.cross_entropy(y_hat, y.argmax(dim=1))
        if torch.isnan(loss).any():
            print("Bug!")
        return loss

    def compute_metrics(self, y, y_hat):
        softmax = F.softmax(y_hat, dim=1)
        self.train_acc(softmax.argmax(dim=1), y.argmax(dim=1))
        # metrics = {
        #     metric: metric_fn(softmax.argmax(dim=1), y.argmax(dim=1)) for metric, metric_fn in self.metrics.items()
        # }
        # metrics = {'_'.join([k, stage]): per_class_metrics[k][idx] for k in ['f1', 'precision', 'recall'] for idx, stage in enumerate(['w', 'n1', 'n2', 'n3', 'rem'])}
        # metrics.update({k: v.mean() for k, v in per_class_metrics.items()})
        # metrics['accuracy'] = per_class_metrics['accuracy'].mean()
        # metrics['baseline'] = y.mean(dim=0).detach()
        return metrics

    def training_step(self, batch, batch_index):

        X, y, _, _ = batch
        # del batch
        y_hat = self.forward(X)
        if torch.isnan(y_hat).any():
            print('Bug!')
        loss = self.compute_loss(y, y_hat)
        metrics = {'_'.join(['train', k]): v for k, v in self.compute_metrics(y, y_hat).items()}

        # logs = {'train_loss': loss, 'train_acc': acc, 'train_baseline': baseline}

        return {'loss': loss, 'log': {**dict(train_loss=loss), **metrics}}

    def validation_step(self, batch, batch_index):

        X, y, _, _ = batch
        # del batch
        y_hat = self.forward(X)
        loss = self.compute_loss(y, y_hat)
        metrics = {'_'.join(['eval', k]): v for k, v in self.compute_metrics(y, y_hat).items()}

        return {**dict(eval_loss=loss), **metrics}
        # return {'eval_loss': loss}.update({'_'.join(['eval', k]): v for k, v in metrics.items()})
        # return {'val_loss': loss, 'val_acc': acc, 'val_baseline': baseline}

    def validation_epoch_end(self, outputs):

        return {'val_loss': torch.stack([x['eval_loss'] for x in outputs]).mean(),
                'log': {k: torch.stack([x[k] for x in outputs]).mean() for k in outputs[0].keys()}}

    def test_step(self, batch, batch_index):

        X, y, current_record, current_sequence = batch
        y_hat = self.forward(X)

        return {'predicted': y_hat.softmax(dim=1), 'true': y, 'record': current_record, 'sequence_nr': current_sequence.cpu().numpy()}

    def test_epoch_end(self, output_results):
        """This method collects the results and sorts the predictions according to record and sequence nr."""
        results = {r: {
                "true": [],
                "true_label": [],
                "predicted": [],
                "predicted_label": [],
                "stable_sleep": [],
            # 'acc': None,
            # 'f1': None,
            # 'recall': None,
            # 'precision': None,
        } for r in self.test_dataloader.dataloader.dataset.records}
        print('Eyoooo')

        for r in self.test_dataloader.dataloader.dataset.records:
            current_record = sorted([v for v in output_results if v['record'][0] == r], key=lambda x: x['sequence_nr'])
            if not current_record:
                results.pop(r, None)
                continue
            y = torch.cat([v['predicted'] for v in current_record], dim=0).permute(1, 0, 2).reshape(self.n_channels, -1)
            t = torch.cat([v['true'] for v in current_record], dim=0).permute(1, 0, 2).reshape(self.n_channels, -1)
            y_label = y.argmax(dim=0)
            t_label = t.argmax(dim=0)
            # cm = ptl.metrics.ConfusionMatrix()(y_label, t_label)
            # acc = ptl.metrics.Accuracy()(y_label, t_label)
            # f1 = ptl.metrics.F1(reduction='none')(y_label, t_label)
            # precision = ptl.metrics.Precision(reduction='none')(y_label, t_label)
            # recall = ptl.metrics.Recall(reduction='none')(y_label, t_label)
            results[r]['true'] = t.cpu().numpy()
            results[r]['true_label'] = t_label.cpu().numpy()
            results[r]['predicted'] = y.cpu().numpy()
            results[r]['predicted_label'] = y_label.cpu().numpy()
            # results[r]['acc'] = acc.cpu().numpy()
            # results[r]['cm'] = cm.cpu().numpy()
            # results[r]['f1'] = f1.cpu().numpy()
            # results[r]['precision'] = precision.cpu().numpy()
            # results[r]['recall'] = recall.cpu().numpy()

        return results

    def configure_optimizers(self):

        # Set common parameters
        self.optimizer_params = {}
        self.trainable_params = [p[1] for p in self.named_parameters() if not 'bias' in p[0] and not 'batch_norm' in p[0]]

        # Change optimizer type and update specific parameters
        if self.hparams.optimizer == "adam":
            self.optimizer_params.update(dict(lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay))
            optimizer = torch.optim.Adam
        elif self.hparams.optimizer == "sgd":
            self.optimizer_params.update(dict(lr=self.hparams.learning_rate, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay))
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError
        optimizer = optimizer(self.trainable_params, **self.optimizer_params)

        # Set scheduler
        if not self.hparams.lr_scheduler:
            self.hparams.lr_scheduler = "reduce_on_plateau"

        if self.hparams.lr_scheduler:
            self.scheduler_params = {}
            if self.hparams.lr_scheduler == "cycliclr":
                self.scheduler_params.update(dict(base_lr=self.hparams.base_lr, max_lr=self.hparams.max_lr, step_size_up=self.hparams.step_size_up))
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.CyclicLR(optimizer, **self.scheduler_params),
                    "interval": "step",
                    "frequency": 1,  # self.steps_per_file,
                    "name": "lr_schedule",
                }
            elif self.hparams.lr_scheduler == "reduce_on_plateau":
                self.scheduler_params.update(dict(factor=self.hparams.lr_reduce_factor, patience=self.hparams.lr_reduce_patience))
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params),
                    'monitor': 'eval_loss',
                    "name": "learning_rate",
                }
            else:
                raise NotImplementedError

            return [optimizer], [scheduler]
        else:
            return {"optimizer": optimizer, "frequency": 1, "interval": "step"}

    def train_dataloader(self):
        """Return training dataloader."""
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True)

    def val_dataloader(self):
        """Return validation dataloader."""
        return torch.utils.data.DataLoader(self.eval_data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True)

    def setup(self, stage):
        if stage == 'fit':
            self.dataset_params = dict(data_dir=self.data_dir, n_jobs=self.n_jobs)
            self.dataset = SscWscPsgDataset(**self.dataset_params)
            self.train_data, self.eval_data = self.dataset.split_data(self.eval_ratio)

    def split_data(self):

        n_total = len(self.dataset)
        n_eval = int(np.ceil(self.eval_ratio * n_total))
        n_train = n_total - n_eval

        self.train_data, self.eval_data = torch.utils.data.random_split(self.dataset, [n_train, n_eval])
        print('Dataset length: ', len(self.dataset))
        print('Train dataset length: ', len(self.train_data))
        print('Eval dataset length: ', len(self.eval_data))

    # def on_post_performance_check(self):
    #     if not self.testing == '1':
    #         self.train_data, self.eval_data = self.dataset.split_data(self.eval_ratio)
    #         # print(self.train_data)
    #         # print(self.eval_data)
    #         print('End of epoch, shuffling training and validation data')
    #         print(self.eval_data)

    @staticmethod
    def add_model_specific_args(parent_parser):

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=True)
        architecture_group = parser.add_argument_group("architecture")
        architecture_group.add_argument("--filter_base", default=4, type=int)
        architecture_group.add_argument("--kernel_size", default=3, type=int)
        architecture_group.add_argument("--max_pooling", default=2, type=int)
        architecture_group.add_argument("--n_blocks", default=7, type=int)
        architecture_group.add_argument("--n_channels", default=5, type=int)
        architecture_group.add_argument("--n_classes", default=5, type=int)
        architecture_group.add_argument("--n_rnn_layers", default=1, type=int)
        architecture_group.add_argument("--n_rnn_units", default=1024, type=int)
        architecture_group.add_argument("--rnn_bidirectional", default=True, action="store_true")
        architecture_group.add_argument("--rnn_dropout", default=0, type=float)

        # OPTIMIZER specific
        optimizer_group.add_argument("--weight_decay", default=0, type=float)

        # LEARNING RATE SCHEDULER specific
        lr_scheduler_group = parser.add_argument_group("lr_scheduler")
        lr_scheduler_group.add_argument("--lr_scheduler", default=None, type=str)
        lr_scheduler_group.add_argument("--base_lr", default=0.05, type=float)
        lr_scheduler_group.add_argument("--lr_reduce_factor", default=0.1, type=float)
        lr_scheduler_group.add_argument("--lr_reduce_patience", default=5, type=int)
        lr_scheduler_group.add_argument("--max_lr", default=0.15, type=float)
        lr_scheduler_group.add_argument("--step_size_up", default=0.05, type=int)

        # DATASET specific
        dataset_group = parser.add_argument_group('dataset')
        dataset_group.add_argument('--data_dir', default='data/train/raw/individual_encodings', type=str)
        dataset_group.add_argument('--eval_ratio', default=0.1, type=float)
        dataset_group.add_argument('--n_jobs', default=-1, type=int)

        # DATALOADER specific
        dataloader_group = parser.add_argument_group('dataloader')
        dataloader_group.add_argument('--batch_size', default=32, type=int)
        dataloader_group.add_argument('--n_workers', default=10, type=int)

        return parser


if __name__ == "__main__":

    model_parameters = dict(
        filter_base=4,
        kernel_size=3,
        max_pooling=2,
        n_blocks=7,
        n_channels=5,
        n_classes=5,
        n_rnn_layers=1,
        n_rnn_units=1024,
        rnn_bidirectional=True,
        rnn_dropout=0,
        optimizer="sgd",
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=0,
        lr_scheduler="cycliclr",
        base_lr=0.05,
        max_lr=0.15,
        step_size_up=0.05,
        # data_dir="data/raw/individual_encodings",
        data_dir="./data/full_length/ssc_wsc/raw/train",
        eval_ratio=0.1,
        n_jobs=-1,
        batch_size=32,
        n_workers=0
    )
    model = MasscModel(**model_parameters)
    model.configure_optimizers()
    model.setup("fit")
    model_summary = ptl.core.memory.ModelSummary(model, "full")
    print(model_summary)

    x_shape = (32, 5, 5 * 60 * 128)
    x = torch.rand(x_shape)
    z = model(x)
    print("z.shape:", z.shape)
    pass
