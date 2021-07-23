from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as ptl

from sklearn import metrics
from tqdm import tqdm

from models import layers


class MASSCAverage(ptl.LightningModule):
    def __init__(
        self, *args, **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(self.hparams.batch_size, self.hparams.n_channels, 5 * 60 * 128)

        # HACK
        if not hasattr(self.hparams, "dilation"):
            self.hparams.dilation = 1
        if not hasattr(self.hparams, "activation"):
            self.hparams.activation = "relu"
        if not hasattr(self.hparams, "no_mix"):
            self.hparams.no_mix = False

# fmt: off
        # Create mixing block
        if self.hparams.n_channels != 1 and not self.hparams.no_mix:
            self.mixing_block = nn.Sequential(OrderedDict([
                ('mix_conv', nn.Conv1d(self.hparams.n_channels, self.hparams.n_channels, 1)),
                ('mix_bn', nn.BatchNorm1d(self.hparams.n_channels)),
                ('mix_relu', nn.ReLU()),
            ]))

        # Create basic block structure
        if self.hparams.block_type == 'residual':
            self.blocks = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    (f'residual_{k}-0', layers.ResidualBlock(
                        self.hparams.n_channels if k == 0 else self.hparams.filter_base * 2 ** (k - 1),
                        self.hparams.filter_base * 2 ** k,
                        self.hparams.kernel_size,
                        strided=True,
                        projection_type='identity'
                    )),
                    (f'residual_{k}-1', layers.ResidualBlock(
                        self.hparams.filter_base * 2 ** k,
                        self.hparams.filter_base * 2 ** k,
                        self.hparams.kernel_size,
                        strided=False,
                        projection_type='identity'
                    ))
                ])) for k in range(self.hparams.n_blocks)
            ])
            classification_in_channels = self.hparams.filter_base * (2 ** (self.hparams.n_blocks - 1))
        elif self.hparams.block_type == 'bottleneck':
            self.blocks = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    (f'bottleneck_{k}', layers.BottleneckBlock(
                        n_filters_in=self.hparams.n_channels if k == 0 else 4 * self.hparams.filter_base * 2 ** (k - 1),
                        n_filters=self.hparams.filter_base * 2 ** k,
                        kernel_size=self.hparams.kernel_size,
                        strided=True
                    ))
                ])) for k in range(self.hparams.n_blocks)
            ])
            classification_in_channels = 4 * self.hparams.filter_base * (2 ** (self.hparams.n_blocks - 1))
        else:
            self.blocks = nn.ModuleList([
                nn.Sequential(OrderedDict([
                    (f'simpleblock_{k}', layers.SimpleBlock(
                        n_filters_in=self.hparams.n_channels if k == 0 else self.hparams.filter_base * 2 ** (k - 1),
                        n_filters=self.hparams.filter_base * 2 ** k,
                        kernel_size=self.hparams.kernel_size,
                        dilation=self.hparams.dilation,
                        activation=self.hparams.activation,
                        strided=True
                    ))
                ])) for k in range(self.hparams.n_blocks)
            ])
            classification_in_channels = self.hparams.filter_base * (2 ** (self.hparams.n_blocks - 1))

        # Create temporal processing block
        if self.hparams.n_rnn_units > 0:
            self.temporal_block = nn.GRU(
                input_size=self.hparams.filter_base * (2 ** (self.hparams.n_blocks - 1)),
                hidden_size=self.hparams.n_rnn_units,
                num_layers=self.hparams.n_rnn_layers,
                batch_first=True,
                dropout=self.hparams.rnn_dropout,
                bidirectional=self.hparams.rnn_bidirectional
            )
            classification_in_channels = (1 + self.hparams.rnn_bidirectional) * self.hparams.n_rnn_units
        else:
            self.temporal_block = None

        # Create classification block
        self.classification = nn.Conv1d(
            in_channels=classification_in_channels,
            out_channels=self.hparams.n_classes,
            kernel_size=1
        )
        # Init bias term to 1 / n_classes
        nn.init.constant_(self.classification.bias, 1 / self.hparams.n_classes)

        # Define loss function
        try:
            self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(self.hparams.cb_weights))
        except AttributeError:
            self.loss = nn.CrossEntropyLoss()


    def forward(self, x):
        if self.temporal_block:
            self.temporal_block.flatten_parameters()

        # Mixing block
        if self.hparams.no_mix:
            z = x
        else:
            z = self.mixing_block(x)

        # Feature extraction block
        for block in self.blocks:
            z = block(z)

        return z

    def classify_segments(self, x, eval_frequency_sec=None):

        if eval_frequency_sec is None:
            eval_frequency_sec = self.hparams.eval_frequency_sec

        # Get 1 s predictions
        z_1 = self(x)

        # Return predicted logits
        logits_1 = self.classification(z_1)

        # Merge with average
        z = z_1.unfold(-1, eval_frequency_sec, eval_frequency_sec).mean(dim=-1)

        if self.temporal_block is not None:
            z = z.permute(0, 2, 1)
            z = self.temporal_block(z)
            z = z[0].permute(0, 2, 1)

        logits = self.classification(z)

        return logits, logits_1

    def compute_loss(self, y_hat, y):
        loss = self.loss(y_hat, y.argmax(dim=1))
        if torch.isnan(loss).any():
            print("Bug!")
        return loss

    # def compute_metrics(self, y, y_hat):
    #     softmax = F.softmax(y_hat, dim=1)
    #     self.train_acc(softmax.argmax(dim=1), y.argmax(dim=1))
    #     return metrics

    def shared_step(self, X, y, stable_sleep):

        if stable_sleep.shape[-1] == 300:
            # Implicitly assuming that stable_sleep is every second
            stable_sleep = stable_sleep[:, :: self.hparams.eval_frequency_sec]
        y = y[:, :, :: self.hparams.eval_frequency_sec]  # Implicitly assuming y is scored every sec.

        y_hat, _ = self.classify_segments(X)
        try:
            loss = self.compute_loss(
                y_hat.transpose(2, 1)[stable_sleep],
                y.transpose(2, 1)[stable_sleep]
            )
        except RuntimeError as err:
            print(f"y: {y.transpose(2, 1)[stable_sleep]}")
            print(f"y.shape: {y.transpose(2, 1)[stable_sleep].shape}")
            print(f"y_hat: {y_hat.transpose(2, 1)[stable_sleep]}")
            print(f"y_hat.shape: {y_hat.transpose(2, 1)[stable_sleep].shape}")
            print(f"stable_sleep: {stable_sleep}")
            print(f"stable_sleep.shape: {stable_sleep.shape}")
            print(err)

        return loss, y_hat, y, stable_sleep

    def training_step(self, batch, batch_index):

        X, y, _, _, stable_sleep = batch
        loss, _, _, _ = self.shared_step(X, y, stable_sleep)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_index):
        X, y, current_record, current_sequence, stable_sleep = batch
        loss, y_hat, y, stable_sleep = self.shared_step(X, y, stable_sleep)
        self.log("eval_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {
            "predicted": y_hat.softmax(dim=1),
            "true": y,
            "record": current_record,
            "sequence_nr": current_sequence,
            "stable_sleep": stable_sleep,
        }

    def validation_epoch_end(self, outputs):

        true = torch.cat([out["true"] for out in outputs], dim=0).permute([0, 2, 1])
        predicted = torch.cat([out["predicted"] for out in outputs], dim=0).permute([0, 2, 1])
        stable_sleep = torch.cat([out["stable_sleep"].to(torch.int64) for out in outputs], dim=0)
        sequence_nrs = torch.cat([out["sequence_nr"] for out in outputs], dim=0)

        if self.use_ddp:
            out_true = [torch.zeros_like(true) for _ in range(torch.distributed.get_world_size())]
            out_predicted = [torch.zeros_like(predicted) for _ in range(torch.distributed.get_world_size())]
            out_stable_sleep = [torch.zeros_like(stable_sleep) for _ in range(torch.distributed.get_world_size())]
            out_seq_nrs = [torch.zeros_like(sequence_nrs) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(out_true, true)
            dist.all_gather(out_predicted, predicted)
            dist.all_gather(out_stable_sleep, stable_sleep)
            dist.all_gather(out_seq_nrs, sequence_nrs)
            # if dist.get_rank() == 0:
            t = (
                torch.stack(out_true)
                .transpose(0, 1)
                .reshape(-1, int(300 / self.hparams.eval_frequency_sec), self.hparams.n_classes)
                .cpu()
                .numpy()
            )
            p = (
                torch.stack(out_predicted)
                .transpose(0, 1)
                .reshape(-1, int(300 / self.hparams.eval_frequency_sec), self.hparams.n_classes)
                .cpu()
                .numpy()
            )
            s = (
                torch.stack(out_stable_sleep)
                .transpose(0, 1)
                .reshape(-1, int(300 / self.hparams.eval_frequency_sec))
                .to(torch.bool)
                .cpu()
                .numpy()
            )
            u = t.sum(axis=-1) == 1

            acc = metrics.accuracy_score(t[s & u].argmax(-1), p[s & u].argmax(-1))
            cohen = metrics.cohen_kappa_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4])
            f1_macro = metrics.f1_score(
                t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average="macro"
            )
            precision = metrics.precision_score(
                t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average=None
            )
            recall = metrics.recall_score(
                t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average=None
            )

            self.log_dict(
                {
                    "eval/acc": acc,
                    "eval/cohen": cohen,
                    "eval_f1_macro": f1_macro,
                    "eval/precision/wake": precision[0],
                    "eval/precision/n1": precision[1],
                    "eval/precision/n2": precision[2],
                    "eval/precision/n3": precision[3],
                    "eval/precision/rem": precision[4],
                    "eval/recall/wake": recall[0],
                    "eval/recall/n1": recall[1],
                    "eval/recall/n2": precision[2],
                    "eval/recall/n3": recall[3],
                    "eval/recall/rem": recall[4],
                },
                prog_bar=False,
                on_epoch=True,
            )

        elif self.on_gpu:
            t = true.cpu().numpy()
            p = predicted.cpu().numpy()
            s = stable_sleep.to(torch.bool).cpu().numpy()
            u = t.sum(axis=-1) == 1

            acc = metrics.accuracy_score(t[s & u].argmax(-1), p[s & u].argmax(-1))
            cohen = metrics.cohen_kappa_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4])
            f1_macro = metrics.f1_score(
                t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average="macro"
            )
            precision = metrics.precision_score(
                t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average=None
            )
            recall = metrics.recall_score(
                t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average=None
            )

            self.log_dict(
                {
                    "eval/acc": acc,
                    "eval/cohen": cohen,
                    "eval/f1_macro": f1_macro,
                    "eval/precision/wake": precision[0],
                    "eval/precision/n1": precision[1],
                    "eval/precision/n2": precision[2],
                    "eval/precision/n3": precision[3],
                    "eval/precision/rem": precision[4],
                    "eval/recall/wake": recall[0],
                    "eval/recall/n1": recall[1],
                    "eval/recall/n2": precision[2],
                    "eval/recall/n3": recall[3],
                    "eval/recall/rem": recall[4],
                },
                prog_bar=True,
                on_epoch=True,
            )
        else:
            t = true.numpy()
            p = predicted.numpy()
            s = stable_sleep.to(torch.bool).numpy()
            u = t.sum(axis=-1) == 1

            acc = metrics.accuracy_score(t[s & u].argmax(-1), p[s & u].argmax(-1))
            cohen = metrics.cohen_kappa_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4])
            f1_macro = metrics.f1_score(
                t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average="macro"
            )

            self.log_dict(
                {"eval/acc": acc, "eval/cohen": cohen, "eval/f1_macro": f1_macro}, prog_bar=True, on_epoch=True
            )

    def test_step(self, batch, batch_index):
        X, y, current_record, current_sequence, stable_sleep = batch
        # stable_sleep = stable_sleep[:, ::self.hparams.eval_frequency_sec]  # Implicitly assuming that stable_sleep is every second
        y = y[:, :, :: self.hparams.eval_frequency_sec]  # Implicitly assuming y is scored every sec.
        # y_hat = self.forward(X)  # TODO: fix this to use classify segments fn

        y_hat, y_hat_1 = self.classify_segments(X)

        # Return predictions at other resolutions
        y_hat_3s, _ = self.classify_segments(X, eval_frequency_sec=3)
        y_hat_5s, _ = self.classify_segments(X, eval_frequency_sec=5)
        y_hat_10s, _ = self.classify_segments(X, eval_frequency_sec=10)
        y_hat_15s, _ = self.classify_segments(X, eval_frequency_sec=15)
        y_hat_60s, _ = self.classify_segments(X, eval_frequency_sec=60)
        y_hat_150s, _ = self.classify_segments(X, eval_frequency_sec=150)
        y_hat_300s, _ = self.classify_segments(X, eval_frequency_sec=300)
        y_hat_600s, _ = self.classify_segments(X, eval_frequency_sec=600)
        y_hat_900s, _ = self.classify_segments(X, eval_frequency_sec=900)
        y_hat_1800s, _ = self.classify_segments(X, eval_frequency_sec=1800)
        y_hat_2700s, _ = self.classify_segments(X, eval_frequency_sec=2700)
        y_hat_3600s, _ = self.classify_segments(X, eval_frequency_sec=3600)
        y_hat_5400s, _ = self.classify_segments(X, eval_frequency_sec=5400)
        y_hat_7200s, _ = self.classify_segments(X, eval_frequency_sec=7200)

        return {
            "predicted": y_hat.softmax(dim=1),
            "true": y,
            "record": current_record,
            "sequence_nr": current_sequence,
            "stable_sleep": stable_sleep,
            "logits": y_hat_1.softmax(dim=1),
            "y_hat_3s": y_hat_3s.softmax(dim=1),
            "y_hat_5s": y_hat_5s.softmax(dim=1),
            "y_hat_10s": y_hat_10s.softmax(dim=1),
            "y_hat_15s": y_hat_15s.softmax(dim=1),
            "y_hat_60s": y_hat_60s.softmax(dim=1),
            "y_hat_150s": y_hat_150s.softmax(dim=1),
            "y_hat_300s": y_hat_300s.softmax(dim=1),
            "y_hat_600s": y_hat_600s.softmax(dim=1),
            "y_hat_900s": y_hat_900s.softmax(dim=1),
            "y_hat_1800s": y_hat_1800s.softmax(dim=1),
            "y_hat_2700s": y_hat_2700s.softmax(dim=1),
            "y_hat_3600s": y_hat_3600s.softmax(dim=1),
            "y_hat_5400s": y_hat_5400s.softmax(dim=1),
            "y_hat_7200s": y_hat_7200s.softmax(dim=1),
            # "data": X.cpu(),
        }

    def test_epoch_end(self, output_results):
        """This method collects the results and sorts the predictions according to record and sequence nr."""

        try:
            all_records = sorted(self.trainer.datamodule.test.records)
        except AttributeError:  # Catch exception if we've supplied dataloaders instead of DataModule
            all_records = sorted(self.trainer.test_dataloaders[0].dataset.records)
        # true = torch.cat([out['true'] for out in output_results], dim=0).permute([0, 2, 1])
        # true = torch.cat([out['true'] for out in output_results], dim=0).transpose(2, 1)
        # predicted = torch.cat([out['predicted'] for out in output_results], dim=0).permute([0, 2, 1])
        # stable_sleep = torch.cat([out['stable_sleep'].to(torch.int64) for out in output_results], dim=0)
        # sequence_nrs = torch.cat([out['sequence_nr'] for out in output_results], dim=0)
        # logits = torch.cat([out['logits'] for out in output_results], dim=-1).permute([0, 2, 1])
        # y_hat_3s = torch.cat([out['y_hat_3s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_5s = torch.cat([out['y_hat_5s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_10s = torch.cat([out['y_hat_10s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_15s = torch.cat([out['y_hat_15s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_60s = torch.cat([out['y_hat_60s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_150s = torch.cat([out['y_hat_150s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_300s = torch.cat([out['y_hat_300s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_600s = torch.cat([out['y_hat_600s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_900s = torch.cat([out['y_hat_900s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_1800s = torch.cat([out['y_hat_1800s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_2700s = torch.cat([out['y_hat_2700s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_3600s = torch.cat([out['y_hat_3600s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_5400s = torch.cat([out['y_hat_5400s'] for out in output_results], dim=0).permute([0, 2, 1])
        # y_hat_7200s = torch.cat([out['y_hat_7200s'] for out in output_results], dim=0).permute([0, 2, 1])
        # data = torch.cat([out['data'] for out in output_results], dim=0).permute([0, 2, 1])
        # records = [i for i, out in enumerate(output_results) for j, _ in enumerate(out['record'])]

        if self.use_ddp:
            raise NotImplementedError
            # records = torch.cat([torch.Tensor([record2int[r]]).type_as(stable_sleep) for out in output_results for r in out['record']], dim=0)
            # out_true = [torch.zeros_like(true) for _ in range(dist.get_world_size())]
            # out_predicted = [torch.zeros_like(predicted) for _ in range(dist.get_world_size())]
            # out_stable_sleep = [torch.zeros_like(stable_sleep) for _ in range(dist.get_world_size())]
            # out_seq_nrs = [torch.zeros_like(sequence_nrs) for _ in range(dist.get_world_size())]
            # out_records = [torch.zeros_like(records) for _ in range(dist.get_world_size())]
            # out_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
            # out_y_hat_3s = [torch.zeros_like(y_hat_3s) for _ in range(dist.get_world_size())]
            # out_y_hat_5s = [torch.zeros_like(y_hat_5s) for _ in range(dist.get_world_size())]
            # out_y_hat_10s = [torch.zeros_like(y_hat_10s) for _ in range(dist.get_world_size())]
            # out_y_hat_15s = [torch.zeros_like(y_hat_15s) for _ in range(dist.get_world_size())]
            # out_y_hat_60s = [torch.zeros_like(y_hat_60s) for _ in range(dist.get_world_size())]
            # out_y_hat_150s = [torch.zeros_like(y_hat_150s) for _ in range(dist.get_world_size())]
            # out_y_hat_300s = [torch.zeros_like(y_hat_300s) for _ in range(dist.get_world_size())]
            # out_y_hat_600s = [torch.zeros_like(y_hat_600s) for _ in range(dist.get_world_size())]
            # out_y_hat_900s = [torch.zeros_like(y_hat_900s) for _ in range(dist.get_world_size())]
            # out_y_hat_1800s = [torch.zeros_like(y_hat_1800s) for _ in range(dist.get_world_size())]
            # out_y_hat_2700s = [torch.zeros_like(y_hat_2700s) for _ in range(dist.get_world_size())]
            # out_y_hat_3600s = [torch.zeros_like(y_hat_3600s) for _ in range(dist.get_world_size())]
            # out_y_hat_5400s = [torch.zeros_like(y_hat_5400s) for _ in range(dist.get_world_size())]
            # out_y_hat_7200s = [torch.zeros_like(y_hat_7200s) for _ in range(dist.get_world_size())]
            # torch.cuda.empty_cache()
            # dist.barrier()
            # dist.all_gather(out_records, records)
            # dist.all_gather(out_predicted, predicted)
            # dist.all_gather(out_true, true)
            # dist.all_gather(out_stable_sleep, stable_sleep)
            # dist.all_gather(out_seq_nrs, sequence_nrs)
            # dist.all_gather(out_logits, logits)
            # dist.all_gather(out_y_hat_3s, y_hat_3s)
            # dist.all_gather(out_y_hat_5s, y_hat_5s)
            # dist.all_gather(out_y_hat_10s, y_hat_10s)
            # dist.all_gather(out_y_hat_15s, y_hat_15s)
            # dist.all_gather(out_y_hat_60s, y_hat_60s)
            # dist.all_gather(out_y_hat_150s, y_hat_150s)
            # dist.all_gather(out_y_hat_300s, y_hat_300s)
            # dist.all_gather(out_y_hat_600s, y_hat_600s)
            # dist.all_gather(out_y_hat_900s, y_hat_900s)
            # dist.all_gather(out_y_hat_1800s, y_hat_1800s)
            # dist.all_gather(out_y_hat_2700s, y_hat_2700s)
            # dist.all_gather(out_y_hat_3600s, y_hat_3600s)
            # dist.all_gather(out_y_hat_5400s, y_hat_5400s)
            # dist.all_gather(out_y_hat_7200s, y_hat_7200s)

            # if dist.get_rank() == 0:
            #     true = torch.cat(out_true)
            #     predicted = torch.cat(out_predicted)
            #     stable_sleep = torch.cat(out_stable_sleep)
            #     sequence_nrs = torch.cat(out_seq_nrs)
            #     records = [int2record[r.item()] for r in torch.cat(out_records)]
            #     logits = torch.cat(out_logits)
            #     y_hat_3s = torch.cat(out_y_hat_3s)
            #     y_hat_5s = torch.cat(out_y_hat_5s)
            #     y_hat_10s = torch.cat(out_y_hat_10s)
            #     y_hat_15s = torch.cat(out_y_hat_15s)
            #     y_hat_60s = torch.cat(out_y_hat_60s)
            #     y_hat_150s = torch.cat(out_y_hat_150s)
            #     y_hat_300s = torch.cat(out_y_hat_300s)
            #     y_hat_600s = torch.cat(out_y_hat_600s)
            #     y_hat_900s = torch.cat(out_y_hat_900s)
            #     y_hat_1800s = torch.cat(out_y_hat_1800s)
            #     y_hat_2700s = torch.cat(out_y_hat_2700s)
            #     y_hat_3600s = torch.cat(out_y_hat_3600s)
            #     y_hat_5400s = torch.cat(out_y_hat_5400s)
            #     y_hat_7200s = torch.cat(out_y_hat_7200s)

            # acc = metrics.accuracy_score(t[s].argmax(-1), p[s].argmax(-1))
            # cohen = metrics.cohen_kappa_score(t[s].argmax(-1), p[s].argmax(-1), labels=[0, 1, 2, 3, 4])
            # f1_macro = metrics.f1_score(t[s].argmax(-1), p[s].argmax(-1), labels=[0, 1, 2, 3, 4], average='macro')

            # self.log_dict({
            #     'eval_acc': acc,
            #     'eval_cohen': cohen,
            #     'eval_f1_macro': f1_macro,
            # }, prog_bar=True)

            # else:
            #     return None
        else:
            records = [r for out in output_results for r in out["record"]]

        results = {
            r: {
                "true": [],
                "true_label": [],
                "predicted": [],
                "predicted_label": [],
                "stable_sleep": [],
                "logits": [],
                "seq_nr": [],
                "yhat_3s": [],
                "yhat_5s": [],
                "yhat_10s": [],
                "yhat_15s": [],
                "yhat_60s": [],
                "yhat_150s": [],
                "yhat_300s": [],
                "yhat_600s": [],
                "yhat_900s": [],
                "yhat_1800s": [],
                "yhat_2700s": [],
                "yhat_3600s": [],
                "yhat_5400s": [],
                "yhat_7200s": [],
            }
            for r in all_records
        }

        for o in tqdm(output_results, desc="Sorting predictions"):
            r = o["record"][0]

            results[r]["true"] = o["true"].squeeze(0).T.cpu().numpy()
            results[r]["predicted"] = o["predicted"].squeeze(0).T.cpu().numpy()
            results[r]["stable_sleep"] = o["stable_sleep"].squeeze(0).cpu().numpy()
            results[r]["logits"] = o["logits"].squeeze(0).T.cpu().numpy()
            results[r]["seq_nr"] = o["sequence_nr"].squeeze(0).cpu().numpy()
            for resolution in [3, 5, 10, 15, 60, 150, 300, 600, 900, 1800, 2700, 3600, 5400, 7200]:
                results[r][f"yhat_{resolution}s"] = o[f"y_hat_{resolution}s"].squeeze(0).T.cpu().numpy()

        # for r in tqdm(all_records, desc='Sorting predictions...'):
        #     record_idx = [idx for idx, rec in enumerate(records) if r == rec]
        #     current_t = true[record_idx].reshape(-1, *true.shape[2:])
        #     current_p = predicted[record_idx].reshape(-1, predicted.shape[-1])
        #     current_ss = stable_sleep[record_idx].reshape(-1).to(torch.bool)
        #     current_l = logits[record_idx].reshape(-1, logits.shape[-1])
        #     current_seq = sequence_nrs[record_idx]
        #     current_3s = y_hat_3s[record_idx].reshape(-1, y_hat_3s.shape[-1])
        #     current_5s = y_hat_5s[record_idx].reshape(-1, y_hat_5s.shape[-1])
        #     current_10s = y_hat_10s[record_idx].reshape(-1, y_hat_10s.shape[-1])
        #     current_15s = y_hat_15s[record_idx].reshape(-1, y_hat_15s.shape[-1])
        #     current_60s = y_hat_60s[record_idx].reshape(-1, y_hat_60s.shape[-1])
        #     current_150s = y_hat_150s[record_idx].reshape(-1, y_hat_150s.shape[-1])
        #     current_300s = y_hat_300s[record_idx].reshape(-1, y_hat_300s.shape[-1])
        #     current_600s = y_hat_600s[record_idx].reshape(-1, y_hat_600s.shape[-1])
        #     current_900s = y_hat_900s[record_idx].reshape(-1, y_hat_900s.shape[-1])
        #     current_1800s = y_hat_1800s[record_idx].reshape(-1, y_hat_1800s.shape[-1])
        #     current_2700s = y_hat_2700s[record_idx].reshape(-1, y_hat_2700s.shape[-1])
        #     current_3600s = y_hat_3600s[record_idx].reshape(-1, y_hat_3600s.shape[-1])
        #     current_5400s = y_hat_5400s[record_idx].reshape(-1, y_hat_5400s.shape[-1])
        #     current_7200s = y_hat_7200s[record_idx].reshape(-1, y_hat_7200s.shape[-1])

        # current_t = torch.cat([t for idx, t in enumerate(true) if idx in record_idx], dim=0)
        # current_p = torch.cat([p for idx, p in enumerate(predicted) if idx in record_idx], dim=0)
        # current_ss = torch.cat([ss.to(torch.bool) for idx, ss in enumerate(stable_sleep) if idx in record_idx], dim=0)
        # current_l = torch.cat([l for idx, l in enumerate(logits) if idx in record_idx], dim=0)
        # current_seq = torch.stack([s for idx, s in enumerate(sequence_nrs) if idx in record_idx])

        # results[r]['true'] = current_t.cpu().numpy()
        # results[r]['predicted'] = current_p.cpu().numpy()
        # results[r]['stable_sleep'] = current_ss.cpu().numpy()
        # results[r]['logits'] = current_l.cpu().numpy()
        # results[r]['seq_nr'] = current_seq.cpu().numpy()
        # results[r]['yhat_3s'] = current_3s.cpu().numpy()
        # results[r]['yhat_5s'] = current_5s.cpu().numpy()
        # results[r]['yhat_10s'] = current_10s.cpu().numpy()
        # results[r]['yhat_15s'] = current_15s.cpu().numpy()
        # results[r]['yhat_60s'] = current_60s.cpu().numpy()
        # results[r]['yhat_150s'] = current_150s.cpu().numpy()
        # results[r]['yhat_300s'] = current_300s.cpu().numpy()
        # results[r]['yhat_600s'] = current_600s.cpu().numpy()
        # results[r]['yhat_900s'] = current_900s.cpu().numpy()
        # results[r]['yhat_1800s'] = current_1800s.cpu().numpy()
        # results[r]['yhat_2700s'] = current_2700s.cpu().numpy()
        # results[r]['yhat_3600s'] = current_3600s.cpu().numpy()
        # results[r]['yhat_5400s'] = current_5400s.cpu().numpy()
        # results[r]['yhat_7200s'] = current_7200s.cpu().numpy()

        return results

    def configure_optimizers(self):

        # Set common parameters
        self.optimizer_params = {}
        self.trainable_params = self.parameters()

        # Change optimizer type and update specific parameters
        if self.hparams.optimizer == "adam":
            self.optimizer_params.update(dict(lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay))
            optimizer = torch.optim.Adam
        elif self.hparams.optimizer == "sgd":
            self.optimizer_params.update(
                dict(
                    lr=self.hparams.learning_rate,
                    momentum=self.hparams.momentum,
                    weight_decay=self.hparams.weight_decay,
                )
            )
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError
        optimizer = optimizer(self.trainable_params, **self.optimizer_params)

        # Set scheduler
        # if not self.hparams.lr_scheduler:
        #     self.hparams.lr_scheduler = "reduce_on_plateau"

        if self.hparams.lr_scheduler:
            self.scheduler_params = {}
            if self.hparams.lr_scheduler == "cycliclr":
                self.scheduler_params.update(
                    dict(
                        base_lr=self.hparams.base_lr, max_lr=self.hparams.max_lr, step_size_up=self.hparams.step_size_up
                    )
                )
                if self.hparams.optimizer == "adam":
                    self.scheduler_params["cycle_momentum"] = False
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.CyclicLR(optimizer, **self.scheduler_params),
                    "interval": "step",
                    "frequency": 1,  # self.steps_per_file,
                    "name": "learning_rate",
                }
            elif self.hparams.lr_scheduler == "reduce_on_plateau":
                self.scheduler_params.update(
                    dict(factor=self.hparams.lr_reduce_factor, patience=self.hparams.lr_reduce_patience)
                )
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params),
                    "monitor": "eval_loss",
                    "name": "learning_rate",
                }
            else:
                raise NotImplementedError

            return [optimizer], [scheduler]
        else:
            return {"optimizer": optimizer, "frequency": 1, "interval": "step"}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    @staticmethod
    def add_model_specific_args(parent_parser):

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=True)
        architecture_group = parser.add_argument_group("architecture")
        architecture_group.add_argument("--filter_base", default=4, type=int)
        architecture_group.add_argument("--kernel_size", default=3, type=int)
        architecture_group.add_argument("--n_blocks", default=7, type=int)
        architecture_group.add_argument("--n_channels", default=5, type=int)
        architecture_group.add_argument("--n_classes", default=5, type=int)
        architecture_group.add_argument("--n_rnn_layers", default=0, type=int)
        architecture_group.add_argument("--n_rnn_units", default=0, type=int)
        architecture_group.add_argument("--rnn_bidirectional", action="store_true")
        architecture_group.add_argument("--rnn_dropout", default=0, type=float)
        architecture_group.add_argument("--loss_weight", default=None, nargs="+", type=float)
        architecture_group.add_argument("--eval_frequency_sec", default=30, type=int)
        architecture_group.add_argument("--n_attention_units", default=256, type=int)
        architecture_group.add_argument("--block_type", type=str, default="simple")
        architecture_group.add_argument("--dilation", type=int, default=1)
        architecture_group.add_argument("--activation", type=str, default="relu")
        architecture_group.add_argument("--no_mix", action="store_true")

        # OPTIMIZER specific
        optimizer_group = parser.add_argument_group("optimizer")
        optimizer_group.add_argument("--optimizer", default="adam", type=str)
        optimizer_group.add_argument("--learning_rate", default=1e-4, type=float)
        optimizer_group.add_argument("--momentum", default=0.9, type=float)
        optimizer_group.add_argument("--weight_decay", default=0, type=float)

        # LEARNING RATE SCHEDULER specific
        lr_scheduler_group = parser.add_argument_group("lr_scheduler")
        lr_scheduler_group.add_argument("--lr_scheduler", default=None, type=str)
        lr_scheduler_group.add_argument("--base_lr", default=0.05, type=float)
        lr_scheduler_group.add_argument("--lr_reduce_factor", default=0.1, type=float)
        lr_scheduler_group.add_argument("--lr_reduce_patience", default=5, type=int)
        lr_scheduler_group.add_argument("--max_lr", default=0.15, type=float)
        lr_scheduler_group.add_argument("--step_size_up", default=0.05, type=int)

        return parser


if __name__ == "__main__":
    from datasets import SscWscDataModule

    parser = ArgumentParser(add_help=False)
    parser = SscWscDataModule.add_dataset_specific_args(parser)
    parser = MASSCAverage.add_model_specific_args(parser)
    args = parser.parse_args()
    model = MASSCAverage(vars(args))
    model_summary = ptl.core.memory.ModelSummary(model, "full")
    print(model_summary)
    x_shape = (32, 5, 5 * 60 * 128)
    x = torch.rand(x_shape)
    z = model.classify_segments(x, 30)
    try:
        print("z.shape:", z.shape)
    except:
        print("z.shape:", z[0].shape)
        print("z.shape:", z[1].shape)
    pass
