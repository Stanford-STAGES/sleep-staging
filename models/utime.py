from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
#from pytorch_lightning import EvalResult, TrainResult
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy

from losses import DiceLoss
from utils.plotting import plot_segment


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, kernel_size=3, dilation=1, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True,
            ),
            nn.ReLU(),
            # nn.ELU(),
            nn.BatchNorm1d(self.out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, filters=[16, 32, 64, 128], in_channels=5, maxpool_kernels=[10, 8, 6, 4], kernel_size=5, dilation=2):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernels = maxpool_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation
        assert len(self.filters) == len(
            self.maxpool_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied maxpool kernels ({len(self.maxpool_kernels)})!"

        self.depth = len(self.filters)

        # fmt: off
        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
        ) for k in range(self.depth)])
        # fmt: on

        self.maxpools = nn.ModuleList([nn.MaxPool1d(self.maxpool_kernels[k]) for k in range(self.depth)])

    def forward(self, x):
        shortcuts = []
        for layer, maxpool in zip(self.blocks, self.maxpools):
            x = layer(x)
            shortcuts.append(x)
            x = maxpool(x)
        return x, shortcuts


class Decoder(nn.Module):
    def __init__(self, filters=[128, 64, 32, 16], upsample_kernels=[4, 6, 8, 10], in_channels=256, out_channels=5, kernel_size=5):
        super().__init__()
        self.filters = filters
        self.upsample_kernels = upsample_kernels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        assert len(self.filters) == len(
            self.upsample_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied upsample kernels ({len(self.upsample_kernels)})!"
        self.depth = len(self.filters)

        # fmt: off
        self.upsamples = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=self.upsample_kernels[k]),
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                activation='relu',
            )
        ) for k in range(self.depth)])

        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
        ) for k in range(self.depth)])
        # fmt: off
        self.dense = nn.Sequential(
            nn.Conv1d(in_channels=self.filters[-1], out_channels=self.out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, z, shortcuts):

        for upsample, block, shortcut in zip(self.upsamples, self.blocks, shortcuts[::-1]):
            z = upsample(z)
            z = torch.cat([shortcut, z], dim=1)
            z = block(z)

        return self.dense(z)


class SegmentClassifier(nn.Module):
    def __init__(self, sampling_frequency=128, num_classes=5, epoch_length=30):
        super().__init__()
        # self.sampling_frequency = sampling_frequency
        self.num_classes = num_classes
        # self.epoch_length = epoch_length
        self.layers = nn.Sequential(
            # nn.AvgPool2d(kernel_size=(1, self.epoch_length * self.sampling_frequency)),
            # nn.Flatten(start_dim=2),
            # nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # batch_size, num_classes, n_samples = x.shape
        # z = x.reshape((batch_size, num_classes, -1, self.epoch_length * self.sampling_frequency))
        return self.layers(x)


class UTimeModel(LightningModule):
    # def __init__(
    #     self, filters=[16, 32, 64, 128], in_channels=5, maxpool_kernels=[10, 8, 6, 4], kernel_size=5,
    #     dilation=2, sampling_frequency=128, num_classes=5, epoch_length=30, lr=1e-4, batch_size=12,
    #     n_workers=0, eval_ratio=0.1, data_dir=None, n_jobs=-1, n_records=-1, scaling=None, **kwargs
    # ):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        # self.save_hyperparameters(hparams)
        self.save_hyperparameters({k: v for k, v in hparams.items() if not callable(v)})
        self.encoder = Encoder(
            filters=self.hparams.filters,
            in_channels=self.hparams.in_channels,
            maxpool_kernels=self.hparams.maxpool_kernels,
            kernel_size=self.hparams.kernel_size,
            dilation=self.hparams.dilation,
        )
        self.bottom = nn.Sequential(
            ConvBNReLU(
                in_channels=self.hparams.filters[-1],
                out_channels=self.hparams.filters[-1] * 2,
                kernel_size=self.hparams.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.hparams.filters[-1] * 2,
                out_channels=self.hparams.filters[-1] * 2,
                kernel_size=self.hparams.kernel_size
            ),
        )
        self.decoder = Decoder(
            filters=self.hparams.filters[::-1],
            upsample_kernels=self.hparams.maxpool_kernels[::-1],
            in_channels=self.hparams.filters[-1] * 2,
            kernel_size=self.hparams.kernel_size,
        )
        self.segment_classifier = SegmentClassifier(
            sampling_frequency=self.hparams.sampling_frequency,
            num_classes=self.hparams.num_classes,
            epoch_length=self.hparams.epoch_length
        )
        self.loss = DiceLoss(self.hparams.num_classes)
        # self.loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.eval_acc = Accuracy()
#        self.metric = Accuracy(num_classes=self.hparams.num_classes, reduce_op='mean')

        # Create Dataset params
        self.dataset_params = dict(
            data_dir=self.hparams.data_dir,
            n_jobs=self.hparams.n_jobs,
            n_records=self.hparams.n_records,
            scaling=self.hparams.scaling,
        )

        # # Create DataLoader params
        # self.dataloader_params = dict(
        #     batch_size=self.hparams.batch_size,
        #     num_workers=self.hparams.n_workers,
        #     pin_memory=True,
        # )

        # Create Optimizer params
        self.optimizer_params = dict(lr=self.hparams.lr)
        # self.example_input_array = torch.zeros(1, self.hparams.in_channels, 35 * 30 * 100)

    def forward(self, x):

        # Run through encoder
        z, shortcuts = self.encoder(x)

        # Bottom section
        z = self.bottom(z)

        # Run through decoder
        z = self.decoder(z, shortcuts)

        return z

    def classify_segments(self, x, resolution=30):

        # Run through encoder + decoder
        z = self(x)

        # Classify decoded samples
        resolution_samples = self.hparams.sampling_frequency * resolution
        z = z.unfold(-1, resolution_samples, resolution_samples) \
             .mean(dim=-1)
        y = self.segment_classifier(z)

        return y

    def training_step(self, batch, batch_idx):
        # if batch_idx == 100:
        #     print('hej')
        x, t, _, _, stable_sleep = batch
        ss = stable_sleep[:, ::30]

        # Classify segments
        y = self.classify_segments(x)

        # loss = self.loss(y, t[:, :, ::self.hparams.epoch_length])
        loss = self.compute_loss(y, t, stable_sleep)
        t = t[:, :, ::30]
        self.train_acc(y.argmax(dim=1)[ss], t.argmax(dim=1)[ss])
        # accuracy = self.metric(y.argmax(dim=1), t[:, :, ::self.hparams.epoch_length].argmax(dim=1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
#        result = TrainResult(minimize=loss)
#        result.log('train_loss', loss, prog_bar=True, sync_dist=True)
#        result.log('train_acc', accuracy, prog_bar=True, sync_dist=True)
#        return result

    def validation_step(self, batch, batch_idx):
        x, t, r, seqs, stable_sleep = batch
        ss = stable_sleep[:, ::30]

        # Classify segments
        y = self.classify_segments(x)

        # loss = self.loss(y, t[:, :, ::self.hparams.epoch_length])
        loss = self.compute_loss(y, t, stable_sleep)
        t = t[:, :, ::30]
        self.eval_acc(y.argmax(dim=1)[ss], t.argmax(dim=1)[ss])
        # accuracy = self.metric(y.argmax(dim=1), t[:, :, ::self.hparams.epoch_length].argmax(dim=1))
        self.log('eval_loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log('eval_acc', self.eval_acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        # # Generate an image
        # if batch_idx == 0:
        #     fig = plot_segment(x, t, z)
        #     self.logger.experiment[1].log({'Hypnodensity': wandb.Image(fig)})
        #     plt.close(fig)

#        result = EvalResult(checkpoint_on=loss)
#        result.log('eval_loss', loss, prog_bar=True, sync_dist=True)
#        result.log('eval_acc', accuracy, prog_bar=True, sync_dist=True)
#
#        return result

    def test_step(self, batch, batch_index):

        X, y, current_record, current_sequence, stable_sleep = batch
        y_hat = self.classify_segments(X)
        # result = ptl.EvalResult()
        # result.predicted = y_hat.softmax(dim=1)
        # result.true = y
        # result.record = current_record
        # result.sequence_nr = current_sequence
        # result.stable_sleep = stable_sleep
        # return result
        return {
            "predicted": y_hat.softmax(dim=1),
            "true": y,
            "record": current_record,
            "sequence_nr": current_sequence,
            "stable_sleep": stable_sleep,
        }

    def test_epoch_end(self, output_results):
        """This method collects the results and sorts the predictions according to record and sequence nr."""
        try:
            all_records = self.test_dataloader.dataloader.dataset.records # When running a new dataset
        except AttributeError:
            all_records = self.test_data.records # When running on the train data
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
            } for r in all_records
        }

        for r in all_records:
            if isinstance(output_results, dict):
                current_record = {k: v for k, v in output_results.items() if k is not 'meta'}
                current_record = [dict(zip(current_record, t)) for t in zip(*current_record.values())]
                current_record = sorted([v for v in current_record if v["record"][0] == r], key=lambda x: x["sequence_nr"])
            elif isinstance(output_results, list):
            # current_record = {k: [record for record in records if ] for k, v in output_results.items() if k is not 'meta'}
            # current_record = {k: v for k, v in current_record.items()}
                current_record = sorted([v for v in output_results if v["record"][0] == r], key=lambda x: x["sequence_nr"])
            if not current_record:
                results.pop(r, None)
                continue
            y = torch.cat([v["predicted"] for v in current_record], dim=0).permute(1, 0, 2).reshape(self.hparams.num_classes, -1)
            try:
                t = torch.cat([v["true"] for v in current_record], dim=0).permute(1, 0, 2).reshape(self.hparams.num_classes, -1)
            except RuntimeError as e:
                raise RuntimeError(f'Current record: {current_record[0]["record"][0]}, {e}')
            stable_sleep = torch.cat([v["stable_sleep"] for v in current_record], dim=0).flatten()
            # y_label = y.argmax(dim=0)
            # t_label = t.argmax(dim=0)
            # cm = ptl.metrics.ConfusionMatrix()(y_label, t_label)
            # acc = ptl.metrics.Accuracy()(y_label, t_label)
            # f1 = ptl.metrics.F1(reduction='none')(y_label, t_label)
            # precision = ptl.metrics.Precision(reduction='none')(y_label, t_label)
            # recall = ptl.metrics.Recall(reduction='none')(y_label, t_label)
            results[r]["true"] = t.cpu().numpy()
            # results[r]["true_label"] = t_label.cpu().numpy()
            results[r]["predicted"] = y.cpu().numpy()
            results[r]['stable_sleep'] = stable_sleep.cpu().numpy()
            # results[r]["predicted_label"] = y_label.cpu().numpy()
            # results[r]['acc'] = acc.cpu().numpy()
            # results[r]['cm'] = cm.cpu().numpy()
            # results[r]['f1'] = f1.cpu().numpy()
            # results[r]['precision'] = precision.cpu().numpy()
            # results[r]['recall'] = recall.cpu().numpy()

            # results_dir = os.path.dirname(self.hparams.resume_from_checkpoint)
            # with open(os.path.join(results_dir, f"{_name}_predictions.pkl"), "wb") as pkl:
            #     pickle.dump(predictions, pkl)

            # df_results = evaluate_performance(predictions)
            # df_results.to_csv(os.path.join(results_dir, f"{_name}_results.csv"))
        # output_results.results = results

        # return output_results
        return results

    def compute_loss(self, y_pred, y_true, stable_sleep):
        stable_sleep = stable_sleep[:, ::self.hparams.epoch_length]
        y_true = y_true[:, :, ::self.hparams.epoch_length]

        if y_pred.shape[-1] != self.hparams.num_classes:
            y_pred = y_pred.permute(dims=[0, 2, 1])
        if y_true.shape[-1] != self.hparams.num_classes:
            y_true = y_true.permute(dims=[0, 2, 1])
        # return self.loss(y_pred, y_true.argmax(dim=-1))

        return self.loss(y_pred, y_true, stable_sleep)

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': list(self.encoder.parameters())},
            {'params': list(self.bottom.parameters())},
            {'params': list(self.decoder.parameters())},
            # {'params': [p[1] for p in self.named_parameters() if 'bias' not in p[0] and 'batch_norm' not in p[0]]},
            {'params': list(self.segment_classifier.parameters())[0], 'weight_decay': 1e-5},
            {'params': list(self.segment_classifier.parameters())[1]},
        ], **self.optimizer_params
        )

    # def on_after_backward(self):
    #     print('Hej')

    # def train_dataloader(self):
    #     """Return training dataloader."""
    #     return DataLoader(self.train_data, shuffle=True, **self.dataloader_params)

    # def val_dataloader(self):
    #     """Return validation dataloader."""
    #     return DataLoader(self.eval_data, shuffle=False, **self.dataloader_params)

    # def setup(self, stage):
    #     if stage == 'fit':
    #         self.dataset = SscWscPsgDataset(**self.dataset_params)
    #         self.train_data, self.eval_data = self.dataset.split_data(self.hparams.eval_ratio)

    @staticmethod
    def add_model_specific_args(parent_parser):

        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=True)
        architecture_group = parser.add_argument_group('architecture')
        architecture_group.add_argument('--filters', default=[16, 32, 64, 128], nargs='+', type=int)
        architecture_group.add_argument('--in_channels', default=5, type=int)
        architecture_group.add_argument('--maxpool_kernels', default=[10, 8, 6, 4], nargs='+', type=int)
        architecture_group.add_argument('--kernel_size', default=5, type=int)
        architecture_group.add_argument('--dilation', default=2, type=int)
        architecture_group.add_argument('--sampling_frequency', default=128, type=int)
        architecture_group.add_argument('--num_classes', default=5, type=int)
        architecture_group.add_argument('--epoch_length', default=30, type=int)

        # OPTIMIZER specific
        optimizer_group = parser.add_argument_group('optimizer')
        # optimizer_group.add_argument('--optimizer', default='sgd', type=str)
        optimizer_group.add_argument('--lr', default=5e-6, type=float)
        # optimizer_group.add_argument('--momentum', default=0.9, type=float)
        # optimizer_group.add_argument('--weight_decay', default=0, type=float)

        # LEARNING RATE SCHEDULER specific
        # lr_scheduler_group = parser.add_argument_group('lr_scheduler')
        # lr_scheduler_group.add_argument('--lr_scheduler', default=None, type=str)
        # lr_scheduler_group.add_argument('--base_lr', default=0.05, type=float)
        # lr_scheduler_group.add_argument('--lr_reduce_factor', default=0.1, type=float)
        # lr_scheduler_group.add_argument('--lr_reduce_patience', default=5, type=int)
        # lr_scheduler_group.add_argument('--max_lr', default=0.15, type=float)
        # lr_scheduler_group.add_argument('--step_size_up', default=0.05, type=int)

        # DATASET specific
        # dataset_group = parser.add_argument_group('dataset')
        # dataset_group.add_argument('--data_dir', default='data/train/raw/individual_encodings', type=str)
        # dataset_group.add_argument('--eval_ratio', default=0.1, type=float)
        # dataset_group.add_argument('--n_jobs', default=-1, type=int)
        # dataset_group.add_argument('--n_records', default=-1, type=int)
        # dataset_group.add_argument('--scaling', default=None, type=str)


        # DATALOADER specific
        # dataloader_group = parser.add_argument_group('dataloader')
        # dataloader_group.add_argument('--batch_size', default=12, type=int)
        # dataloader_group.add_argument('--n_workers', default=0, type=int)

        return parser


if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)
    parser = UTimeModel.add_model_specific_args(parser)
    args = parser.parse_args()
    # parser.add_argument('--filters', default=[16, 32, 64, 128], nargs='+', type=int)
    # args = parser.parse_args()
    # print('Filters:', args.filters)
    in_channels = args.in_channels
    x_shape = (12, in_channels, 10 * 30 * 128)
    x = torch.rand(x_shape)

    # # Test ConvBNReLU block
    # z = ConvBNReLU()(x)
    # print()
    # print(ConvBNReLU())
    # print(x.shape)
    # print(z.shape)

    # # test Encoder class
    # encoder = Encoder()
    # print(encoder)
    # print("x.shape:", x.shape)
    # z, shortcuts = encoder(x)
    # print("z.shape:", z.shape)
    # print("Shortcuts shape:", [shortcut.shape for shortcut in shortcuts])

    # # Test Decoder class
    # z_shape = (32, 256, 54)
    # z = torch.rand(z_shape)
    # decoder = Decoder()
    # print(decoder)
    # x_hat = decoder(z, None)
    # print("x_hat.shape:", x_hat.shape)

    # Test UTimeModel Class
    # utime = UTimeModel(in_channels=in_channels)
    utime = UTimeModel(**vars(args))
    utime.configure_optimizers()
    print(utime)
    print(x.shape)
    # z = utime(x)
    z = utime.classify_segments(x)
    print(z.shape)
    print("x.shape:", x.shape)
    print("z.shape:", z.shape)
    print(z.sum(dim=1))
