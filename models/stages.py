import logging
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as ptl
import sklearn.metrics as metrics

# from pytorch_lightning.metrics import Accuracy
# from pytorch_lightning.metrics import F1
# from pytorch_lightning.metrics import Precision
# from pytorch_lightning.metrics import Recall

logger = logging.getLogger("lightning")


# fmt: off
class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = [
            self.kernel_size[1] // 2 - self.stride[1] % 2 if self.stride[1] > 1 else self.kernel_size[1] // 2, self.kernel_size[1] // 2,
            self.kernel_size[2] // 2, self.kernel_size[2] // 2,
            0, 0
        ]
        # self.padding = [
        #     self.kernel_size[1] // 2 - self.kernel_size[1] % 2 + (1 - self.stride[1] % 2), self.kernel_size[1] // 2,
        #     self.kernel_size[2] // 2 - self.kernel_size[2] % 2, self.kernel_size[2] // 2,
        #     0, 0
        # ]

        self.layers = nn.Sequential(OrderedDict([
            ('padd', nn.ConstantPad3d(self.padding, 0.0)),
            ("conv", nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride)),
            ("batch_norm", nn.BatchNorm3d(num_features=out_channels)),
            ("relu", nn.ReLU()),
        ]))

        # Initialize weights according to old code
        nn.init.normal_(self.layers[1].weight, 1e-3)
        nn.init.constant_(self.layers[1].bias, 0.0)

    # def _calculate_padding(ks, s, d=1):
    #     if s % 2
    #     return

    def forward(self, x):
        return self.layers(x)


class MaxPool3D(nn.Module):
    def __init__(self, kernel_size=None, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = [
            self.kernel_size[1] // 2 - self.stride[1] % 2 if self.stride[1] > 1 else self.kernel_size[1] // 2, self.kernel_size[1] // 2,
            self.kernel_size[2] // 2, self.kernel_size[2] // 2,
            0, 0
        ]

        self.layers = nn.Sequential(OrderedDict([
            ("padd", nn.ConstantPad3d(self.padding, 0.0)),
            ("maxpool", nn.MaxPool3d(kernel_size=self.kernel_size, stride=self.stride))
        ]))

    def forward(self, x):
        return self.layers(x)


class LargeAutocorr(nn.Module):
    def __init__(self, model_name, modality, seg_size):
        super().__init__()
        self.model_name = model_name
        self.modality = modality
        self.seg_size = seg_size

        if modality == "eeg":
            np.random.seed(int(self.model_name[-2:]) + 1)
            self.n_in = 2
            self.n_out = [64, 128, 256, 512]
            self.strides = [[3, 2], [2, 1]]
        elif modality == "eog":
            np.random.seed(int(self.model_name[-2:]) + 2)
            self.n_in = 3
            self.n_out = [64, 128, 256, 512]
            self.strides = [[4, 2], [2, 1]]
        elif modality == "emg":
            np.random.seed(int(self.model_name[-2:]) + 3)
            self.n_in = 1
            self.n_out = [16, 32, 64, 128]
            self.strides = [[2, 2], [2, 1]]
        else:
            raise ValueError(f"Please specify modality, got {modality}.")

        self.layers = nn.Sequential(OrderedDict([
            (f"conv-01-{modality}", Conv2DBlock(self.n_in, self.n_out[0], [1, 7, 7], [1] + self.strides[0])),
            (f"maxpool-01-{modality}", MaxPool3D(kernel_size=[1, 3, 3], stride=[1, 2, 2])),
            # (f"maxpool-01-{modality}", nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 0, 1])),
            (f"conv-02-{modality}", Conv2DBlock(self.n_out[0], self.n_out[1], [1, 5, 5], [1] + self.strides[1])),
            (f"conv-03-{modality}", Conv2DBlock(self.n_out[1], self.n_out[1], [1, 5, 5], [1, 1, 1])),
            (f"maxpool-02-{modality}", MaxPool3D(kernel_size=[1, 2, 2], stride=[1, 2, 2])),
            # (f"maxpool-02-{modality}", nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 1, 1])),
            (f"conv-04-{modality}", Conv2DBlock(self.n_out[1], self.n_out[2], [1, 3, 3], [1, 1, 1])),
            (f"conv-05-{modality}", Conv2DBlock(self.n_out[2], self.n_out[2], [1, 3, 3], [1, 1, 1])),
            (f"maxpool-03-{modality}", nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])),
            # (f"maxpool-03-{modality}", nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 1, 0])),
            (f"conv-06-{modality}", Conv2DBlock(self.n_out[2], self.n_out[3], [1, 3, 3], [1, 1, 1])),
            (f"conv-07-{modality}", Conv2DBlock(self.n_out[3], self.n_out[3], [1, 3, 3], [1, 1, 1])),
        ]))

    def forward(self, x):
        batch_size, n_features, n_segs, seg_size = x.shape
        z = x.reshape(batch_size, self.n_in, -1, n_segs, seg_size).permute(0, 1, 3, 2, 4)
        for layer in self.layers:
            z = layer(z)
        z = torch.mean(z, dim=-1)
        z = torch.mean(z, dim=-1)
        return z


class RandomAutocorr(torch.nn.Module):
    def __init__(self, model_name, modality, seg_size):
        super().__init__()
        self.model_name = model_name
        self.modality = modality
        self.seg_size = seg_size

        if modality == "eeg":
            np.random.seed(int(self.model_name[-2:]) + 1)
            self.n_in = 2
            self.n_out = [np.random.randint(32, 96), np.random.randint(64, 192), np.random.randint(128, 384)]
            self.strides = [[3, 2], [2, 1]]
        elif modality == "eog":
            np.random.seed(int(self.model_name[-2:]) + 2)
            self.n_in = 3
            self.n_out = [np.random.randint(32, 96), np.random.randint(64, 192), np.random.randint(128, 384)]
            self.strides = [[4, 2], [2, 1]]
        elif modality == "emg":
            np.random.seed(int(self.model_name[-2:]) + 3)
            self.n_in = 1
            self.n_out = [np.random.randint(8, 24), np.random.randint(16, 48), np.random.randint(32, 96)]
            self.strides = [[2, 2], [2, 1]]
        else:
            raise ValueError(f"Please specify modality, got {modality}.")

        self.layers = nn.Sequential(OrderedDict([
            (f"conv-01-{modality}", Conv2DBlock(self.n_in, self.n_out[0], [1, 7, 7], [1] + self.strides[0])),
            (f"maxpool-01-{modality}", nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])),
            (f"conv-02-{modality}", Conv2DBlock(self.n_out[0], self.n_out[1], [1, 5, 5], [1] + self.strides[1])),
            (f"conv-03-{modality}", Conv2DBlock(self.n_out[1], self.n_out[1], [1, 3, 3], [1, 1, 1])),
            (f"maxpool-02-{modality}", nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 1, 1])),
            (f"conv-04-{modality}", Conv2DBlock(self.n_out[1], self.n_out[2], [1, 3, 3], [1, 1, 1])),
            (f"conv-05-{modality}", Conv2DBlock(self.n_out[2], self.n_out[2], [1, 3, 3], [1, 1, 1])),
        ]))

    def forward(self, x):
        batch_size, n_features, n_segs, seg_size = x.shape
        # batch_size, n_segs, seg_size, n_features = x.shape
        z = x.reshape(batch_size, self.n_in, -1, n_segs, seg_size).permute(0, 1, 3, 2, 4)
        # z = x.reshape(batch_size, n_segs, seg_size, self.n_in, -1).permute(0, 3, 1, 4, 2)
        for layer in self.layers:
            z = layer(z)
        z = torch.mean(z, dim=-1)
        z = torch.mean(z, dim=-1)
        return z
# fmt: on


class StagesModel(ptl.LightningModule):

    # def __init__(self, batch_size=5, data_dir=None, dropout=None, eval_ratio=0.1, learning_rate=0.005, model_name=None,
    #              momentum=0.9, n_classes=None, n_hidden_units=None, n_jobs=None, n_workers=None, seg_size=None, **kwargs):
    def __init__(
        self,
        # hparams,
        batch_size=None,
        dropout=None,
        model_name=None,
        n_classes=None,
        n_hidden_units=None,
        seg_size=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        # self.save_hyperparameters({k: v for k, v in hparams.items() if not callable(v)})
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(
            self.hparams.batch_size, 400 + 1200 + 40, int(1200 / self.hparams.seg_size), self.hparams.seg_size
        )
        # self.__dict__.update(kwargs)
        # self.hparams = kwargs
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

        self.hparams.gamma = np.exp(-1 / 750)
        self.hparams.lstm = self.hparams.model_name.split("_")[3] == "lstm"
        self.hparams.eval_frequency = 15 if self.hparams.model_name.split("_")[2] == "ls" else 30
        self.hparams.steps_per_file = 1  # 300 // self.batch_size

        if self.hparams.model_name.split("_")[1] == "rh":
            self.hidden_eeg = RandomAutocorr(self.hparams.model_name, "eeg", self.hparams.seg_size)
            self.hidden_eog = RandomAutocorr(self.hparams.model_name, "eog", self.hparams.seg_size)
            self.hidden_emg = RandomAutocorr(self.hparams.model_name, "emg", self.hparams.seg_size)
        elif self.hparams.model_name.split("_")[1] == "lh":
            self.hidden_eeg = LargeAutocorr(self.hparams.model_name, "eeg", self.hparams.seg_size)
            self.hidden_eog = LargeAutocorr(self.hparams.model_name, "eog", self.hparams.seg_size)
            self.hidden_emg = LargeAutocorr(self.hparams.model_name, "emg", self.hparams.seg_size)
        self.hparams.n_out_eeg = self.hidden_eeg.n_out
        self.hparams.n_out_eog = self.hidden_eog.n_out
        self.hparams.n_out_emg = self.hidden_emg.n_out

        self.hparams.n_hidden_features = (
            self.hidden_eeg.layers[-1].layers[1].weight.shape[1]
            + self.hidden_eog.layers[-1].layers[1].weight.shape[1]
            + self.hidden_emg.layers[-1].layers[1].weight.shape[1]
        )

        if self.hparams.lstm:
            self.hidden_hidden = nn.ModuleList(
                [
                    nn.Dropout(p=1 - self.hparams.dropout),
                    torch.nn.LSTM(
                        self.hparams.n_hidden_features, self.hparams.n_hidden_units, bias=True, batch_first=True
                    ),
                    nn.Dropout(p=1 - self.hparams.dropout),
                ]
            )

        # Classification layer
        self.output_layer = nn.Linear(self.hparams.n_hidden_units, self.hparams.n_classes, bias=True)
        nn.init.normal_(self.output_layer.weight, std=0.04)
        nn.init.constant_(self.output_layer.bias, 0.001)

    def forward(self, x):
        if self.hparams.lstm:
            self.hidden_hidden[1].flatten_parameters()

        x_eeg, x_eog, x_emg = torch.split(x, [400, 1200, 40], dim=1)

        # Feature extraction
        z_eeg = self.hidden_eeg(x_eeg)
        z_eog = self.hidden_eog(x_eog)
        z_emg = self.hidden_emg(x_emg)
        z = torch.cat([z_eeg, z_eog, z_emg], dim=1).permute([0, 2, 1])

        # Feature processing
        if self.hparams.lstm:
            z = self.hidden_hidden[0](z)  # Dropout on input
            z, _ = self.hidden_hidden[1](z)  # LSTM
            z = self.hidden_hidden[2](z)  # Dropout on output
        else:
            raise NotImplementedError

        # Output layer
        # z = z.reshape([-1, self.hparams.n_hidden_units])
        logits = self.output_layer(z)

        return logits

    def compute_loss(self, y_hat, y, w=None):
        y = y.mean(dim=-1).permute([0, 2, 1]).reshape(-1, self.hparams.n_classes)
        y_hat = y_hat.reshape(-1, self.hparams.n_classes)
        if w is not None:
            w = w.mean(dim=-1).reshape(-1)
            loss = F.cross_entropy(torch.clamp(y_hat, min=-1e10, max=1e10), y.argmax(dim=1), reduction="none")
            loss = loss @ w / w.sum()
        else:
            loss = F.cross_entropy(torch.clamp(y_hat, min=-1e10, max=1e10), y.argmax(dim=1))
        if torch.isnan(loss).any():
            print("Bug!")
        return loss

    def compute_metrics(self, y_hat, y):
        y = y.mean(dim=-1).permute([0, 2, 1]).reshape(-1, self.hparams.n_classes)
        softmax = F.softmax(y_hat, dim=1)
        metrics = {"accuracy": (softmax.argmax(dim=-1) == y.argmax(dim=-1)).to(torch.float32).mean()}
        # per_class_metrics = {metric: metric_fn(softmax.argmax(dim=1), y.argmax(dim=1)) for metric, metric_fn in self.metrics.items()}
        # metrics = {'_'.join([k, stage]): per_class_metrics[k][idx] for k in ['f1', 'precision', 'recall'] for idx, stage in enumerate(['w', 'n1', 'n2', 'n3', 'rem'])}
        # metrics.update({k: v.mean() for k, v in per_class_metrics.items()})
        # metrics['accuracy'] = per_class_metrics['accuracy'].mean()
        # metrics['baseline'] = y.mean(dim=0).detach()
        return metrics

    def shared_step(self, X, y, w):

        y_hat = self.forward(X)
        if torch.isnan(y_hat).any():
            logger.warning("Bug!")
        loss = self.compute_loss(y_hat, y, w)
        # metrics = {"_".join(["train", k]): v for k, v in self.compute_metrics(y, y_hat).items()}
        return loss, y_hat

    def training_step(self, batch, batch_index):

        X, y, w, _, _ = batch
        batch_size, n_features, seg_len = X.shape
        X = X.unfold(-1, self.hparams.seg_size, self.hparams.seg_size)
        y = y.unfold(-1, self.hparams.seg_size, self.hparams.seg_size)
        w = w.unfold(-1, self.hparams.seg_size, self.hparams.seg_size)
        loss, _ = self.shared_step(X, y, w)
        # del batch
        # if torch.isnan(X).any():
        #     print('Hej')
        # X[torch.isnan(X)] = torch.tensor(2 ** -23, device=self.device) # Hack to fix zero division from encoding
        # y_hat = self.forward(X)
        if torch.isnan(loss).any():
            print("Bug!")
        # loss = self.compute_loss(y_hat, y)
        # metrics = {"_".join(["train", k]): v for k, v in self.compute_metrics(y, y_hat).items()}

        # logs = {'train_loss': loss, 'train_acc': acc, 'train_baseline': baseline}

        # return {"loss": loss, "log": {**dict(train_loss=loss), **metrics}}
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_index):
        X, y, w, current_record, current_sequences = batch
        batch_size, n_features, seg_len = X.shape
        X = X.unfold(-1, self.hparams.seg_size, self.hparams.seg_size)
        y = y.unfold(-1, self.hparams.seg_size, self.hparams.seg_size)
        w = w.unfold(-1, self.hparams.seg_size, self.hparams.seg_size)
        loss, y_hat = self.shared_step(X, y, w)
        self.log("eval_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {
            "predicted": y_hat.softmax(dim=-1),
            "true": y.mean(dim=-1).permute([0, 2, 1]),
            "record": current_record,
            "sequence_nr": current_sequences,
        }
        # X, y, w = batch
        # del batch
        # y_hat = self.forward(X)
        # loss = self.compute_loss(y, y_hat)
        # metrics = {"_".join(["eval", k]): v for k, v in self.compute_metrics(y, y_hat).items()}

        # return {**dict(eval_loss=loss), **metrics}
        # return {'eval_loss': loss}.update({'_'.join(['eval', k]): v for k, v in metrics.items()})
        # return {'val_loss': loss, 'val_acc': acc, 'val_baseline': baseline}
        # return {"predicted": y_hat.softmax(dim=1), "true": y}

    def validation_epoch_end(self, outputs):

        true = torch.cat([out["true"] for out in outputs], dim=0)
        predicted = torch.cat([out["predicted"] for out in outputs], dim=0)
        sequence_nrs = torch.cat([out["sequence_nr"] for out in outputs], dim=0)

        if self.use_ddp:
            out_true = [torch.zeros_like(true) for _ in range(torch.distributed.get_world_size())]
            out_predicted = [torch.zeros_like(predicted) for _ in range(torch.distributed.get_world_size())]
            out_seq_nrs = [torch.zeros_like(sequence_nrs) for _ in range(dist.get_world_size())]
            dist.barrier()
            dist.all_gather(out_true, true)
            dist.all_gather(out_predicted, predicted)
            dist.all_gather(out_seq_nrs, sequence_nrs)
            if dist.get_rank() == 0:
                t = torch.cat(out_true).cpu().numpy()
                p = torch.cat(out_predicted).cpu().numpy()
                u = t.sum(axis=-1) == 1

                acc = metrics.accuracy_score(t[u].argmax(-1), p[u].argmax(-1))
                cohen = metrics.cohen_kappa_score(t[u].argmax(-1), p[u].argmax(-1), labels=[0, 1, 2, 3, 4])
                f1_macro = metrics.f1_score(t[u].argmax(-1), p[u].argmax(-1), labels=[0, 1, 2, 3, 4], average="macro")

                self.log_dict(
                    {"eval_acc": acc, "eval_cohen": cohen, "eval_f1_macro": f1_macro}, prog_bar=True, on_epoch=True
                )
        elif self.on_gpu:
            t = true.cpu().numpy()
            p = predicted.cpu().numpy()
            u = t.sum(axis=-1) == 1

            acc = metrics.accuracy_score(t[u].argmax(-1), p[u].argmax(-1))
            cohen = metrics.cohen_kappa_score(t[u].argmax(-1), p[u].argmax(-1), labels=[0, 1, 2, 3, 4])
            f1_macro = metrics.f1_score(t[u].argmax(-1), p[u].argmax(-1), labels=[0, 1, 2, 3, 4], average="macro")

            self.log_dict(
                {"eval_acc": acc, "eval_cohen": cohen, "eval_f1_macro": f1_macro}, prog_bar=True, on_epoch=True
            )
        else:
            t = true.numpy()
            p = predicted.numpy()
            u = t.sum(axis=-1) == 1

            acc = metrics.accuracy_score(t[u].argmax(-1), p[u].argmax(-1))
            cohen = metrics.cohen_kappa_score(t[u].argmax(-1), p[u].argmax(-1), labels=[0, 1, 2, 3, 4])
            f1_macro = metrics.f1_score(t[u].argmax(-1), p[u].argmax(-1), labels=[0, 1, 2, 3, 4], average="macro")

            self.log_dict(
                {"eval_acc": acc, "eval_cohen": cohen, "eval_f1_macro": f1_macro}, prog_bar=True, on_epoch=True
            )

    def test_step(self, batch, batch_index):

        X, y, current_record, current_sequence, stable_sleep = batch
        stable_sleep = stable_sleep[
            :, :: self.hparams.eval_frequency_sec
        ]  # Implicitly assuming that stable_sleep is every second
        y = y[:, :, :: self.hparams.eval_frequency_sec]  # Implicitly assuming y is scored every sec.
        # y_hat = self.forward(X)  # TODO: fix this to use classify segments fn
        y_hat, alphas, y_hat_1 = self.classify_segments(X, eval_frequency_sec=1)
        return {
            "predicted": y_hat.softmax(dim=1),
            "true": y,
            "record": current_record,
            "sequence_nr": current_sequence,
            "stable_sleep": stable_sleep,
            "logits": y_hat_1.softmax(dim=1),
            "alphas": alphas.reshape(alphas.shape[0], -1),
        }

    def test_epoch_end(self, output_results):
        """This method collects the results and sorts the predictions according to record and sequence nr."""

        try:
            all_records = sorted(self.trainer.datamodule.test.records)
        except AttributeError:  # Catch exception if we've supplied dataloaders instead of DataModule
            all_records = sorted(self.trainer.test_dataloaders[0].dataset.records)
        true = torch.cat([out["true"] for out in output_results], dim=0).permute([0, 2, 1])
        predicted = torch.cat([out["predicted"] for out in output_results], dim=0).permute([0, 2, 1])
        stable_sleep = torch.cat([out["stable_sleep"].to(torch.int64) for out in output_results], dim=0)
        sequence_nrs = torch.cat([out["sequence_nr"] for out in output_results], dim=0)
        # records = [i for i, out in enumerate(output_results) for j, _ in enumerate(out['record'])]

        if self.use_ddp:
            record2int = {r: idx for idx, r in enumerate(all_records)}
            int2record = {idx: r for idx, r in enumerate(all_records)}
            records = torch.cat(
                [torch.Tensor([record2int[r]]).type_as(stable_sleep) for out in output_results for r in out["record"]],
                dim=0,
            )
            out_true = [torch.zeros_like(true) for _ in range(dist.get_world_size())]
            out_predicted = [torch.zeros_like(predicted) for _ in range(dist.get_world_size())]
            out_stable_sleep = [torch.zeros_like(stable_sleep) for _ in range(dist.get_world_size())]
            out_seq_nrs = [torch.zeros_like(sequence_nrs) for _ in range(dist.get_world_size())]
            out_records = [torch.zeros_like(records) for _ in range(dist.get_world_size())]
            out_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
            out_alphas = [torch.zeros_like(alphas) for _ in range(dist.get_world_size())]

            dist.barrier()
            dist.all_gather(out_true, true)
            dist.all_gather(out_predicted, predicted)
            dist.all_gather(out_stable_sleep, stable_sleep)
            dist.all_gather(out_seq_nrs, sequence_nrs)
            dist.all_gather(out_records, records)
            dist.all_gather(out_logits, logits)
            dist.all_gather(out_alphas, alphas)

            if dist.get_rank() == 0:
                true = torch.cat(out_true)
                predicted = torch.cat(out_predicted)
                stable_sleep = torch.cat(out_stable_sleep)
                sequence_nrs = torch.cat(out_seq_nrs)
                records = [int2record[r.item()] for r in torch.cat(out_records)]
                logits = torch.cat(out_logits)
                alphas = torch.cat(out_alphas)

                # acc = metrics.accuracy_score(t[s].argmax(-1), p[s].argmax(-1))
                # cohen = metrics.cohen_kappa_score(t[s].argmax(-1), p[s].argmax(-1), labels=[0, 1, 2, 3, 4])
                # f1_macro = metrics.f1_score(t[s].argmax(-1), p[s].argmax(-1), labels=[0, 1, 2, 3, 4], average='macro')

                # self.log_dict({
                #     'eval_acc': acc,
                #     'eval_cohen': cohen,
                #     'eval_f1_macro': f1_macro,
                # }, prog_bar=True)

            else:
                return None
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
                "alphas": [],
                # 'acc': None,
                # 'f1': None,
                # 'recall': None,
                # 'precision': None,
            }
            for r in all_records
        }

        for r in tqdm(all_records, desc="Sorting predictions..."):
            record_idx = [idx for idx, rec in enumerate(records) if r == rec]
            current_t = torch.cat([t for idx, t in enumerate(true) if idx in record_idx], dim=0)
            current_p = torch.cat([p for idx, p in enumerate(predicted) if idx in record_idx], dim=0)
            current_ss = torch.cat(
                [ss.to(torch.bool) for idx, ss in enumerate(stable_sleep) if idx in record_idx], dim=0
            )
            current_l = torch.cat([l for idx, l in enumerate(logits) if idx in record_idx], dim=0)
            current_a = torch.cat([a for idx, a in enumerate(alphas) if idx in record_idx], dim=0)

            results[r]["true"] = current_t.cpu().numpy()
            results[r]["predicted"] = current_p.cpu().numpy()
            results[r]["stable_sleep"] = current_ss.cpu().numpy()
            results[r]["logits"] = current_l.cpu().numpy()
            results[r]["alphas"] = current_a.cpu().numpy()

        return results

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(
            [
                {
                    "params": [
                        p[1] for p in self.named_parameters() if "bias" not in p[0] and "batch_norm" not in p[0]
                    ],
                    "weight_decay": self.hparams.weight_decay,
                },
                {"params": [p[1] for p in self.named_parameters() if "bias" in p[0] or "batch_norm" in p[0]]},
            ],
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, self.hparams.gamma),
            "interval": "epoch",
            "frequency": 1,  # self.steps_per_file,
            "name": "lr_schedule",
        }
        # scheduler = None
        # return optimizer
        return [optimizer], [scheduler]

    # def collate_fn(self, batch):

    #     X, y, w = [torch.FloatTensor(b) for b in batch]

    #     return X.squeeze(), y.squeeze(), w.squeeze()

    # def train_dataloader(self):
    #     """Return training dataloader. Batching is handled in the __getitem__ of the dataset."""
    #     return torch.utils.data.DataLoader(self.train_data, batch_size=None, shuffle=False, num_workers=self.n_workers, pin_memory=True)
    #     # return torch.utils.data.DataLoader(self.train_data, batch_size=None, collate_fn=self.collate_fn, shuffle=False, num_workers=self.n_workers, pin_memory=True)

    # def val_dataloader(self):
    #     """Return validation dataloader. Batching is handled in the __getitem__ of the dataset."""
    #     # return torch.utils.data.DataLoader(self.eval_data, batch_size=None, collate_fn=self.collate_fn, shuffle=False, num_workers=self.n_workers, pin_memory=True)
    #     return torch.utils.data.DataLoader(self.eval_data, batch_size=None, shuffle=False, num_workers=self.n_workers, pin_memory=True)

    # def prepare_data(self):
    # print('Hej')
    # self.dataset_params = dict(
    #     data_dir=self.data_dir, n_classes=self.n_classes, n_jobs=self.n_jobs, seg_size=self.seg_size)
    # self.dataset = StagesData(**self.dataset_params)
    # self.split_data()
    # self.train_data, self.eval_data = self.dataset.split_data(0.1)
    # print(self.train_data)
    # print(self.eval_data)

    # def setup(self, stage):
    #     print('Setup')
    #     self.dataset_params = dict(
    #         data_dir=self.data_dir, n_classes=self.n_classes, n_jobs=self.n_jobs, seg_size=self.seg_size)
    #     self.dataset = StagesData(**self.dataset_params)
    #     self.train_data, self.eval_data = self.dataset.split_data(self.eval_ratio)

    # def split_data(self):

    #     n_total = len(self.dataset)
    #     n_eval = int(np.ceil(self.eval_ratio * n_total))
    #     n_train = n_total - n_eval

    #     self.train_data, self.eval_data = torch.utils.data.random_split(self.dataset, [
    #                                                                     n_train, n_eval])
    #     print('Dataset length: ', len(self.dataset))
    #     print('Train dataset length: ', len(self.train_data))
    #     print('Eval dataset length: ', len(self.eval_data))

    # def on_post_performance_check(self):
    #     self.train_data, self.eval_data = self.dataset.split_data(self.eval_ratio)
    #     print(self.train_data)
    #     print(self.eval_data)

    @staticmethod
    def add_model_specific_args(parent_parser):

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--batch_size', default=5, type=int)
        parser.add_argument("--dropout", default=0.5, type=float)
        parser.add_argument("--model_name", default="ac_rh_ls_lstm_01", type=str)
        parser.add_argument("--n_hidden_units", default=128, type=int)
        parser.add_argument("--n_classes", default=5, type=int)
        parser.add_argument("--seg_size", default=60, type=int)

        # OPTIMIZER specific
        parser.add_argument("--learning_rate", default=0.005, type=float)
        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--weight_decay", default=1e-5, type=float)

        # # Data specific
        # parser.add_argument('--data_dir', default='data/train_data', type=str)
        # parser.add_argument('--eval_ratio', default=0.1, type=float)
        # parser.add_argument('--n_jobs', default=-1, type=int)
        # parser.add_argument('--n_workers', default=10, type=int)

        return parser


if __name__ == "__main__":

    from datasets import STAGESDataModule

    parser = ArgumentParser(add_help=False)
    parser = STAGESDataModule.add_dataset_specific_args(parser)
    parser = StagesModel.add_model_specific_args(parser)
    args = parser.parse_args()
    args.n_records = 10

    model = StagesModel(vars(args))
    model.configure_optimizers()
    # model.configure_optimizers()
    # model.prepare_data()
    model_summary = ptl.core.memory.ModelSummary(model, "full")
    print(model_summary)
    # x = torch.rand(model.example_input_array.shape)
    # print(x.shape)

    dm = STAGESDataModule(**vars(args))
    dm.setup()
    train_loader = dm.train_dataloader()

    batch = next(iter(train_loader))

    z = model.training_step(x, None)
    print("z.shape: ", z.shape)
    # seg_size = 60
    # n_segs = 20
    # batch_size = 5
    # n_features = 400 + 1200 + 40
    # x = torch.rand([batch_size, n_segs, seg_size, n_features])
    # z = model(x)
    # print("z.shape:", z.shape)
    # pass
