import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from pytorch_lightning import LightningModule
from scipy.special import softmax
from sklearn import metrics
from tqdm import tqdm


activation_fns = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'tanh': nn.Tanh,
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, dilation=None, activation=None, padding='same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation_fns[activation]
        self.padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2

        self.layers = nn.Sequential(
            nn.ZeroPad2d((self.padding, self.padding, 0, 0)),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=(1, self.kernel_size),
                dilation=(1, self.dilation),
                bias=False,
            ),
            self.activation(),
            nn.BatchNorm2d(self.out_channels)
        )
        nn.init.xavier_uniform_(self.layers[1].weight)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, activation=None, base_filter=None, in_channels=None, maxpool_kernel=None, kernel_size=None, dilation=None, depth=None, complexity_factor=None):
        super().__init__()
        assert all(v is not None for v in [activation, base_filter, in_channels, maxpool_kernel, kernel_size, dilation, depth, complexity_factor])
        self.base_filter = base_filter
        self.in_channels = in_channels
        self.maxpool_kernel = maxpool_kernel
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.depth = depth
        self.activation = activation
        self.complexity_factor = np.sqrt(complexity_factor)
        self.filters = self.__calc_filters()
        # fmt: off
        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBlock(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation=self.activation,
            )
        ) for k in range(self.depth)])
        # fmt: on

        self.maxpool = nn.MaxPool2d((1, self.maxpool_kernel))
        self.zeropad = nn.ZeroPad2d((1, 0, 0, 0))

        self.bottom = ConvBlock(
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=self.kernel_size,
            dilation=1,
            activation=self.activation
        )

    def __calc_filters(self):
        f = [self.base_filter]
        [f.append(int(f[-1] * np.sqrt(2))) for _ in range(self.depth)]
        return [int(v * self.complexity_factor) for v in f]

    def forward(self, x):
        shortcuts = []
        for encoder_block in self.blocks:
            z = encoder_block(x)
            if z.shape[-1] % 2:
                z = self.zeropad(z)
            shortcuts.append(z)
            x = self.maxpool(z)
            # print('x.shape:', x.shape)

        # Bottom part
        encoded = self.bottom(x)

        return encoded, shortcuts


class Decoder(nn.Module):
    def __init__(self, activation=None, filters=None, kernel_size=None, upsample_kernel=None):
        super().__init__()
        assert all(v is not None for v in [activation, filters, kernel_size, upsample_kernel])
        self.activation = activation
        self.filters = filters
        self.kernel_size = kernel_size
        self.upsample_kernel = upsample_kernel
        self.depth = len(filters) - 1

        # fmt: off
        self.upsample_blocks = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=(1, self.upsample_kernel)),
            ConvBlock(
                in_channels=self.filters[k],
                out_channels=self.filters[k + 1],
                kernel_size=self.kernel_size,
                dilation=1,
                activation=self.activation
            )
        ) for k in range(self.depth)])

        self.conv_blocks = nn.ModuleList([nn.Sequential(
            ConvBlock(
                in_channels=self.filters[k + 1] * 2,
                out_channels=self.filters[k + 1],
                kernel_size=self.kernel_size,
                dilation=1,
                activation=self.activation
            )
        ) for k in range(self.depth)])
        # fmt: off

    def _maybe_crop(self, tensor_a, tensor_b):
        s_a, s_b = tensor_a.shape[2:], tensor_b.shape[2:]
        if s_a[-1] == s_b[-1]:
            cropped_tensor_a = tensor_a
        else:
            c = s_a[-1] - s_b[-1]
            c = (c // 2, c // 2 + c % 2)
            cropped_tensor_a = tensor_a[:, :, :, c[0]:-c[1]]
        return cropped_tensor_a

    def forward(self, z, shortcuts):

        for upsample_block, conv_block, shortcut in zip(self.upsample_blocks, self.conv_blocks, shortcuts):
            z = upsample_block(z)
            # print(z.shape)
            # print(shortcut.shape)
            z = self._maybe_crop(z, shortcut)
            z = torch.cat([shortcut, z], dim=1)
            z = conv_block(z)

        return z


class Dense(nn.Module):
    def __init__(self, activation=None, in_channels=None, out_channels=None):
        super().__init__()
        self.activation = activation_fns[activation]
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dense = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=(1, 1),
            ),
            self.activation(),
        )
        nn.init.xavier_uniform_(self.dense[0].weight)
        nn.init.zeros_(self.dense[0].bias)

    def forward(self, x):
        return self.dense(x)


class SegmentClassifier(nn.Module):
    def __init__(self, activation=None, in_channels=None, num_classes=None):
        super().__init__()
        self.activation = activation_fns[activation]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_classes, kernel_size=1),
            self.activation(),
            nn.Conv2d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=1),
        )
        nn.init.xavier_normal_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)
        nn.init.xavier_normal_(self.layers[2].weight)
        nn.init.zeros_(self.layers[2].bias)

    def forward(self, x):
        return self.layers(x)


class USleepModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        hp = self.hparams
        self.example_input_array = torch.zeros(hp.batch_size, hp.n_channels, 1, hp.sequence_length * 60 * 128)

        self.encoder = Encoder(
            activation=hp.activation,
            base_filter=hp.base_filter,
            complexity_factor=hp.complexity_factor,
            depth=hp.depth,
            dilation=hp.dilation,
            in_channels=hp.n_channels,
            kernel_size=hp.kernel_size,
            maxpool_kernel=hp.maxpool,
        )
        self.decoder = Decoder(
            activation=hp.activation,
            filters=self.encoder.filters[::-1],
            kernel_size=hp.kernel_size,
            upsample_kernel=hp.upsample_kernel,
        )
        self.dense = Dense(activation='tanh', in_channels=self.encoder.filters[0], out_channels=self.encoder.filters[0])
        self.segment_classifier = SegmentClassifier(
            activation=hp.activation,
            in_channels=self.encoder.filters[0],
            num_classes=hp.n_classes
        )

        # if hasattr(hp, 'cb_weights'):
        #     self.loss = nn.CrossEntropyLoss(weight=torch.Tensor(hp.cb_weights), reduction='none')
        # else:
        self.loss = nn.CrossEntropyLoss()

        # Create Optimizer params
        self.optimizer_params = dict(lr=hp.lr)

    def forward(self, x):

        # Run through encoder
        z, shortcuts = self.encoder(x)

        # Run through decoder
        z = self.decoder(z, shortcuts[::-1])

        # Run dense modeling
        z = self.dense(z)

        return z

    def classify_segments(self, x, eval_frequency_sec=None):

        if eval_frequency_sec is None:
            eval_frequency_sec = self.hparams.eval_frequency_sec

        # Run through encoder + decoder
        z = self(x.unsqueeze(-2))

        # Logits for every sample
        logits = self.segment_classifier(z).squeeze(-2)

        # Classify decoded samples
        resolution_samples = int(128 * eval_frequency_sec)
        z = z.unfold(-1, resolution_samples, resolution_samples) \
             .mean(dim=-1)
        y = self.segment_classifier(z).squeeze(-2)

        return y, logits

    def compute_loss(self, y_hat, y):
        loss = self.loss(y_hat, y.argmax(dim=1))
        if torch.isnan(loss).any():
            print('Loss returned NaN')
        return loss

    def shared_step(self, x, y, stable_sleep):
        if stable_sleep.shape[-1] == 300:
            stable_sleep = stable_sleep[:, :: self.hparams.eval_frequency_sec]
        y = y[:, :, :: self.hparams.eval_frequency_sec]

        y_hat, _ = self.classify_segments(x)

        loss = self.compute_loss(
            y_hat.transpose(2, 1)[stable_sleep],
            y.transpose(2, 1)[stable_sleep]
        )

        return loss, y_hat.softmax(1), y, stable_sleep

    def training_step(self, batch, batch_idx):
        X, y, _, _, stable_sleep = batch
        loss, _, _, _ = self.shared_step(X, y, stable_sleep)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, current_record, current_sequence, stable_sleep = batch
        loss, y_hat, y, stable_sleep = self.shared_step(X, y, stable_sleep)
        self.log("eval_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return {
            "predicted": y_hat,
            "true": y,
            "record": current_record,
            "sequence_nr": current_sequence,
            "stable_sleep": stable_sleep,
        }

    def validation_epoch_end(self, outputs):
        true = torch.cat([out["true"] for out in outputs if out['true'].shape[-1] == 2 * self.hparams.sequence_length], dim=0).permute([0, 2, 1])
        predicted = torch.cat([out["predicted"] for out in outputs if out['predicted'].shape[-1] == 2 * self.hparams.sequence_length], dim=0).permute([0, 2, 1])
        stable_sleep = torch.cat([out["stable_sleep"].to(torch.int64) for out in outputs if out['stable_sleep'].shape[-1] == 2 * self.hparams.sequence_length], dim=0)
        sequence_nrs = torch.cat([out["sequence_nr"] for out in outputs if out['sequence_nr'].shape[-1] == self.hparams.sequence_length // 5], dim=0)

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
                .reshape(-1, *true.shape[1:])
                .cpu()
                .numpy()
            )
            p = (
                torch.stack(out_predicted)
                .transpose(0, 1)
                .reshape(-1, *predicted.shape[1:])
                .cpu()
                .numpy()
            )
            s = (
                torch.stack(out_stable_sleep)
                .transpose(0, 1)
                .reshape(-1, *stable_sleep.shape[1:])
                .to(torch.bool)
                .cpu()
                .numpy()
            )
            u = t.sum(axis=-1) == 1

        elif self.on_gpu:
            t = true.cpu().numpy()
            p = predicted.cpu().numpy()
            s = stable_sleep.to(torch.bool).cpu().numpy()
            u = t.sum(axis=-1) == 1

        else:
            t = true.numpy()
            p = predicted.numpy()
            s = stable_sleep.to(torch.bool).numpy()
            u = t.sum(axis=-1) == 1

        acc = metrics.accuracy_score(t[s & u].argmax(-1), p[s & u].argmax(-1))
        cohen = metrics.cohen_kappa_score(t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4])
        f1_macro = metrics.f1_score(
            t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average="macro", zero_division=0
        )
        precision = metrics.precision_score(
            t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average=None, zero_division=0
        )
        recall = metrics.recall_score(
            t[s & u].argmax(-1), p[s & u].argmax(-1), labels=[0, 1, 2, 3, 4], average=None, zero_division=0
        )
        self.log_dict(
            {
                "eval/acc": acc,
                "eval/cohen": cohen,
                "eval/f1_macro": f1_macro,
            },
            prog_bar=True,
            on_epoch=True,
        )
        self.log_dict(
            {
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

    def test_step(self, batch, batch_index):
        x, t, r, seqs, stable_sleep = batch
        t = t[:, :, ::30]

        # Classify segments
        # yhat_30s, yhat_1s = self.classify_segments(x)

        # Return predictions at other resolutions
        resolutions = [1/128, 32/128, 64/128, 96/128, 1, 3, 5, 10, 15, 30, 60, 150, 300, 600, 900, 1800, 2700, 3600, 5400, 7200]
        outputs = {
            f'yhat_{resolution}s': self.classify_segments(x, resolution)[0].softmax(1).squeeze(0).T.cpu().numpy() for resolution in resolutions
        }
        outputs = {
            "predicted": outputs['yhat_30s'],
            "true": t.squeeze(0).T.cpu().numpy(),
            "record": r[0],
            "sequence_nr": seqs.cpu().numpy(),
            "stable_sleep": stable_sleep.squeeze(0).cpu().numpy(),
            "logits": outputs['yhat_1s'],
            **outputs
        }

        # We save directly to disk here
        cohort = self.test_dataloader.dataloader.dataset.data_dir.split(os.path.sep)[1]
        results_dir = os.path.join(os.path.dirname(self.trainer.resume_from_checkpoint), "predictions", cohort)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        with open(os.path.join(results_dir, f"preds_{outputs['record'].split('.')[0]}.pkl"), "wb") as pkl:
            pickle.dump(outputs, pkl)
        # yhat_3s, _ = self.classify_segments(x, resolution=3)

        # return {
        #     # "predicted": yhat_30s.softmax(dim=1),
        #     "predicted": outputs['yhat_30s'].softmax(dim=1),
        #     "true": t,
        #     "record": r,
        #     "sequence_nr": seqs,
        #     "stable_sleep": stable_sleep,
        #     "logits": outputs['yhat_1s'].softmax(dim=1),
        #     **outputs
        # }
            # "y_hat_3s": outputs['yhat_3s'].softmax(dim=1),
        # }.update(outputs)

    # def test_epoch_end(self, output_results):
    #     """This method collects the results and sorts the predictions according to record and sequence nr."""
    #     try:
    #         all_records = sorted(self.trainer.datamodule.test.records)
    #     except AttributeError: # Catch exception if we've supplied dataloaders instead of DataModule
    #         all_records = sorted(self.trainer.test_dataloaders[0].dataset.records)

    #     # true = torch.cat([out['true'] for out in output_results], dim=0).transpose(2, 1)
    #     # predicted = torch.cat([out['predicted'] for out in output_results], dim=0).permute([0, 2, 1])
    #     # stable_sleep = torch.cat([out['stable_sleep'].to(torch.int64) for out in output_results], dim=0)
    #     # sequence_nrs = torch.cat([out['sequence_nr'] for out in output_results], dim=0)
    #     # logits = torch.cat([out['logits'] for out in output_results], dim=0).permute([0, 2, 1])
    #     # y_hat_3s = torch.cat([out['y_hat_3s'] for out in output_results], dim=0).permute([0, 2, 1])

    #     if self.use_ddp:
    #         raise NotImplementedError
    #         # record2int = {r: idx for idx, r in enumerate(all_records)}
    #         # int2record = {idx: r for idx, r in enumerate(all_records)}
    #         # records = torch.cat([torch.Tensor([record2int[r]]).type_as(stable_sleep) for out in output_results for r in out['record']], dim=0)
    #         # out_true = [torch.zeros_like(true) for _ in range(dist.get_world_size())]
    #         # out_predicted = [torch.zeros_like(predicted) for _ in range(dist.get_world_size())]
    #         # out_stable_sleep = [torch.zeros_like(stable_sleep) for _ in range(dist.get_world_size())]
    #         # out_seq_nrs = [torch.zeros_like(sequence_nrs) for _ in range(dist.get_world_size())]
    #         # out_records = [torch.zeros_like(records) for _ in range(dist.get_world_size())]
    #         # out_logits = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
    #         # out_y_hat_3s = [torch.zeros_like(y_hat_3s) for _ in range(dist.get_world_size())]

    #         # dist.barrier()
    #         # dist.all_gather(out_true, true)
    #         # dist.all_gather(out_predicted, predicted)
    #         # dist.all_gather(out_stable_sleep, stable_sleep)
    #         # dist.all_gather(out_seq_nrs, sequence_nrs)
    #         # dist.all_gather(out_records, records)
    #         # dist.all_gather(out_logits, logits)
    #         # dist.all_gather(out_y_hat_3s, y_hat_3s)

    #         # if dist.get_rank() == 0:
    #         #     true = torch.cat(out_true)
    #         #     predicted = torch.cat(out_predicted)
    #         #     stable_sleep = torch.cat(out_stable_sleep)
    #         #     sequence_nrs = torch.cat(out_seq_nrs)
    #         #     records = [int2record[r.item()] for r in torch.cat(out_records)]
    #         #     logits = torch.cat(out_logits)
    #         #     y_hat_3s = torch.cat(out_y_hat_3s)

    #         # else:
    #         #     return None
    #     else:
    #         records = [r for out in output_results for r in out['record']]

    #     results = {
    #         r: {
    #             "true": [],
    #             "true_label": [],
    #             "predicted": [],
    #             "predicted_label": [],
    #             "stable_sleep": [],
    #             "logits": [],
    #             "seq_nr": [],
    #             **{k: [] for k in output_results[0].keys() if 'yhat' in k}
    #             # "yhat_3s": [],

    #         } for r in all_records
    #     }

    #     for o in tqdm(output_results, desc="Sorting predictions"):
    #         r = o["record"][0]

    #         results[r]["true"] = o["true"].squeeze(0).T.cpu().numpy()
    #         results[r]["predicted"] = o["predicted"].squeeze(0).T.cpu().numpy()
    #         results[r]["stable_sleep"] = o["stable_sleep"].squeeze(0).cpu().numpy()
    #         results[r]["logits"] = o["logits"].squeeze(0).T.cpu().numpy()
    #         results[r]["seq_nr"] = o["sequence_nr"].squeeze(0).cpu().numpy()
    #         for resolution_key in [res for res in o.keys() if 'yhat' in res]:
    #             results[r][resolution_key] = o[resolution_key].squeeze(0).T.cpu().numpy()

    #     # for r in tqdm(all_records, desc='Sorting predictions...'):
    #     #     record_idx = [idx for idx, rec in enumerate(records) if r == rec]
    #     #     current_t = true[record_idx].reshape(-1, *true.shape[2:])
    #     #     current_p = predicted[record_idx].reshape(-1, predicted.shape[-1])
    #     #     current_ss = stable_sleep[record_idx].reshape(-1).to(torch.bool)
    #     #     current_l = logits[record_idx].reshape(-1, logits.shape[-1])
    #     #     current_seq = sequence_nrs[record_idx]
    #     #     current_3s = y_hat_3s[record_idx].reshape(-1, y_hat_3s.shape[-1])

    #     #     results[r]['true'] = current_t.cpu().numpy()
    #     #     results[r]['predicted'] = current_p.cpu().numpy()
    #     #     results[r]['stable_sleep'] = current_ss.cpu().numpy()
    #     #     results[r]['logits'] = current_l.cpu().numpy()
    #     #     results[r]['seq_nr'] = current_seq.cpu().numpy()
    #     #     results[r]['yhat_3s'] = current_3s.cpu().numpy()

        # return results

    # def compute_loss(self, y_pred, y_true, stable_sleep):
    #     stable_sleep = stable_sleep[:, ::self.hparams.epoch_length].squeeze()
    #     y_true = y_true[:, :, ::self.hparams.epoch_length]

    #     # if y_pred.shape[-1] != self.hparams.num_classes:
    #     #     y_pred = y_pred.permute(dims=[0, 2, 1])
    #     # if y_true.shape[-1] != self.hparams.num_classes:
    #     #     y_true = y_true.permute(dims=[0, 2, 1])
    #     # return self.loss(y_pred, y_true.argmax(dim=-1))

    #     # return self.loss(y_pred, y_true, stable_sleep)
    #     loss = self.loss(y_pred[stable_sleep], y_true.argmax(dim=1)[stable_sleep])
    #     N, L = loss.shape
    #     return (loss.sum(axis=-1) / L).sum() / N

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), **self.optimizer_params
        )

    @staticmethod
    def add_model_specific_args(parent_parser):

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=True)
        architecture_group = parser.add_argument_group('architecture')
        architecture_group.add_argument('--activation', default='elu', type=str)
        architecture_group.add_argument('--base_filter', default=5, type=int)
        architecture_group.add_argument('--complexity_factor', default=1.67, type=float)
        architecture_group.add_argument('--depth', default=12, type=int)
        architecture_group.add_argument('--dilation', default=1, type=int)
        architecture_group.add_argument('--eval_frequency_sec', default=30, type=int)
        architecture_group.add_argument('--eval_frequencies', default=None, type=float, nargs='+')
        architecture_group.add_argument('--kernel_size', default=9, type=int)
        architecture_group.add_argument('--maxpool', default=2, type=int)
        architecture_group.add_argument('--n_classes', default=5, type=int)
        architecture_group.add_argument('--upsample_kernel', default=2, type=int)

        # OPTIMIZER specific
        optimizer_group = parser.add_argument_group('optimizer')
        optimizer_group.add_argument('--optimizer', default='adam', type=str)
        optimizer_group.add_argument('--lr', default=1e-7, type=float)

        return parser


if __name__ == "__main__":

    from pytorch_lightning.core.memory import ModelSummary

    parser = ArgumentParser(add_help=False)
    parser = USleepModel.add_model_specific_args(parser)
    args = parser.parse_args()
    batch_size = 64
    T = 35
    in_channels = 5
    epoch_length = 30
    fs = 128
    x_shape = (batch_size, in_channels, 1, T * epoch_length * fs)
    x = torch.rand(x_shape)

    # # Test ConvBNReLU block
    # z = ConvBNReLU()(x)
    # print()
    # print(ConvBNReLU())
    # print(x.shape)
    # print(z.shape)

    # test Encoder class
    encoder = Encoder(
        activation='elu',
        base_filter=args.base_filter,
        in_channels=in_channels,
        maxpool_kernel=args.maxpool,
        kernel_size=args.kernel_size,
        dilation=args.dilation,
        depth=args.depth,
        complexity_factor=args.complexity_factor
    )
    print(encoder)
    print("x.shape:", x.shape)
    z, shortcuts = encoder(x)
    print("z.shape:", z.shape)
    print("Shortcuts shape:", [shortcut.shape for shortcut in shortcuts])


    # TEST DECODER
    decoder = Decoder(
        activation='elu',
        filters=encoder.filters[::-1],
        kernel_size=args.kernel_size,
        upsample_kernel=args.upsample_kernel,
    )
    x_hat = decoder(z, shortcuts[::-1])


    # TEST DENSE
    dense = Dense(activation='tanh', in_channels=encoder.filters[0], out_channels=encoder.filters[0])
    x_hat = dense(x_hat)
    print('x_hat.shape:', x_hat.shape)

    # TEST SEQUENCE MODELING
    segment_classifier = SegmentClassifier(activation='elu', in_channels=encoder.filters[0], num_classes=args.n_classes)
    x_hat = segment_classifier(x_hat)
    print('x_hat.shape:', x_hat.shape)

    # TEST USLEEP MODEL
    usleep = USleepModel(batch_size=batch_size, fs=fs, n_channels=in_channels, sequence_length=T * epoch_length, **vars(args))
    model_summary = ModelSummary(usleep, "top")
    print(model_summary)
    model_summary = ModelSummary(usleep, "full")
    print(model_summary)
    z_usleep = usleep(x)
    # # Test Decoder class
    # z_shape = (32, 256, 54)
    # z = torch.rand(z_shape)
    # decoder = Decoder()
    # print(decoder)
    # x_hat = decoder(z, None)
    # print("x_hat.shape:", x_hat.shape)

    # Test UTimeModel Class
    # utime = UTimeModel(in_channels=in_channels)
    # usleep = USleepModel(**vars(args))
    # usleep.configure_optimizers()
    # model_summary = ModelSummary(usleep, "top")
    # print(model_summary)
    # print(usleep)
    # print(x.shape)
    # # z = utime(x)
    # z, z_1 = usleep.classify_segments(x)
    # print(z.shape)
    # print("x.shape:", x.shape)
    # print("z.shape:", z.shape)
    # print(z.sum(dim=1))
