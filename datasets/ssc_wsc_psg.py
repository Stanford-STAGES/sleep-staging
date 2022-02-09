import logging
import os
import random
import warnings
from itertools import compress

import numpy as np
import torch
import pytorch_lightning as pl
from h5py import File
from joblib import delayed
from joblib import Memory
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from utils import ParallelExecutor, load_h5_data, initialize_record
except ImportError:
    from utils.parallel_bar import ParallelExecutor

warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
SCALERS = {"robust": RobustScaler, "standard": StandardScaler}
DEFAULT_SEQUENCE_LEN = 5

logger = logging.getLogger()


class SscWscPsgDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        n_jobs=-1,
        scaling=None,
        adjustment=30,
        n_records=-1,
        overlap=True,
        beta=0.999,
        cv=None,
        cv_idx=None,
        eval_ratio=None,
        balanced_sampling=False,
        sequence_length=5,
        max_eval_records=1000,
        n_channels=5,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.n_jobs = n_jobs
        self.scaling = scaling
        self.adjustment = adjustment
        self.n_records = n_records
        self.overlap = overlap
        self.beta = beta
        self.cv = cv
        self.cv_idx = cv_idx
        self.eval_ratio = eval_ratio
        self.balanced_sampling = balanced_sampling
        self.sequence_length = sequence_length
        self.max_eval_records = max_eval_records
        self.n_channels = n_channels
        self.n_classes = 5
        self.records = sorted(os.listdir(self.data_dir))[: self.n_records]
        self.cache_dir = "data/.cache_"
        memory = Memory(self.cache_dir, mmap_mode="r", verbose=0)
        get_data = memory.cache(initialize_record)
        # get_data = initialize_record

        # Get information about the data
        print(f"Loading mmap data using {n_jobs} workers:")
        sorted_data = ParallelExecutor(n_jobs=n_jobs, prefer="threads")(total=len(self.records))(
            delayed(get_data)(
                filename=os.path.join(self.data_dir, record),
                scaling=self.scaling,
                adjustment=self.adjustment,
                overlap=self.overlap,
                sequence_length=self.sequence_length,
            )
            for record in self.records
        )
        cum_class_counts = np.zeros(self.n_classes, dtype=np.int64)

        # This contains the indices for each record (i.e. index 0 maps to first sequence of recording 1 etc.)
        self.record_indices = dict([s["record_indices"] for s in sorted_data])
        # This holds the sequence indices containing each stage in reach record
        self.record_class_indices = dict([s["record_class_indices"] for s in sorted_data])
        self.scalers = dict([s["scalers"] for s in sorted_data])
        self.stable_sleep = dict([s["stable_sleep"] for s in sorted_data])
        self.index_to_record = [sub for s in sorted_data for sub in s["index_to_record"]]
        cum_class_counts = sum([s["cum_class_counts"][1] for s in sorted_data if len(s["cum_class_counts"][1]) == 5])

        # Define the class-balanced weights. We normalize the class counts to the lowest value as the numerator
        # otherwise will dominate the expression
        # (see https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)
        self.cb_weights_norm = (1 - self.beta) / (1 - self.beta ** (cum_class_counts / cum_class_counts.min()))
        self.effective_samples = 1 / self.cb_weights_norm
        self.cb_weights = self.cb_weights_norm * self.n_classes / self.cb_weights_norm.sum()
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("")
            print(f"Class counts: {cum_class_counts}")
            print(f"Beta: {self.beta}")
            print(f"CB weights norm: {self.cb_weights_norm}")
            print(f"Effective samples: {self.effective_samples}")
            print(f"CB weights: {self.cb_weights}")
            print("")
            print("Finished loading data")

    def shuffle_records(self):
        random.shuffle(self.records)

    def split_data(self):
        n_records = len(self.records)
        self.shuffle_records()

        if self.cv:
            from sklearn.model_selection import KFold, StratifiedKFold

            kf = KFold(n_splits=np.abs(self.cv))

            if self.cv > 0:
                train_idx, eval_idx = list(kf.split(range(n_records)))[self.cv_idx]
            print("\n")
            print(f"Running {np.abs(self.cv)}-fold cross-validation procedure.")
            print(f"Current split: {self.cv_idx}")
            print(f"Eval record indices: {eval_idx}")
            print(f"Train record indices: {train_idx}")
            print(f"Number of train/eval records: {len(train_idx)}/{len(eval_idx)}")
            print("\n")
        else:
            n_eval = min(int(n_records * self.eval_ratio), self.max_eval_records)
            n_train = n_records - n_eval
            train_idx = np.arange(n_eval, n_records)
            eval_idx = np.arange(0, n_eval)

        self.train_idx = train_idx
        self.eval_idx = eval_idx
        self.train_data = SscWscPsgSubset(self, train_idx, balanced_sampling=self.balanced_sampling, name="Train")
        self.eval_data = SscWscPsgSubset(self, eval_idx, name="Validation")

        return self.train_data, self.eval_data

    def __len__(self):
        return len(self.index_to_record)

    def __getitem__(self, idx):

        try:
            current_record = self.index_to_record[idx]["record"]
            # If using balanced sampling, we skip the idx and choose from records directly in the training data
            if self.balanced_sampling and current_record in set(self.train_data.records):
                current_record = np.random.choice(self.train_data.records)
                scaler = self.scalers[current_record]
                while True:
                    class_choice = np.random.choice(list(self.record_class_indices[current_record].keys()))
                    if self.record_class_indices[current_record][class_choice]:
                        current_sequence_idx = np.random.choice(self.record_class_indices[current_record][class_choice])
                        if self.sequence_length != DEFAULT_SEQUENCE_LEN:
                            try:
                                sequence_start = np.random.choice(
                                    np.arange(
                                        np.max(
                                            [
                                                self.record_indices[current_record][0],
                                                current_sequence_idx - 2 * self.sequence_length // DEFAULT_SEQUENCE_LEN + 2,
                                            ]
                                        ),
                                        np.min(
                                            [
                                                self.record_indices[current_record][-1]
                                                - 2 * self.sequence_length // DEFAULT_SEQUENCE_LEN
                                                + 2,
                                                current_sequence_idx,
                                            ]
                                        )
                                        + 1,
                                        2,
                                    )
                                )
                            except:
                                print("")
                            sequence_stop = sequence_start + 2 * self.sequence_length // DEFAULT_SEQUENCE_LEN
                            current_sequence = slice(sequence_start, sequence_stop, 2)
                        else:
                            current_sequence = current_sequence_idx
                        break
            else:
                scaler = self.scalers[current_record]
                if not isinstance(self.sequence_length, str) and self.sequence_length != DEFAULT_SEQUENCE_LEN:
                    sequence_start = self.index_to_record[idx]["idx"]
                    sequence_stop = self.index_to_record[idx]["idx"] + 2 * self.sequence_length // DEFAULT_SEQUENCE_LEN
                    current_sequence = slice(sequence_start, sequence_stop, 2)
                else:
                    current_sequence = self.index_to_record[idx]["idx"]
            stable_sleep = np.array(self.stable_sleep[current_record][current_sequence]).squeeze(-1)

            if isinstance(self.sequence_length, str) and self.sequence_length == "full":
                current_sequence = slice(None)

            # Grab data
            with File(os.path.join(self.data_dir, current_record), "r") as f:
                x = f["M"][current_sequence].astype("float32")
                t = f["L"][current_sequence].astype("uint8").squeeze(-1)

        except IndexError:
            print("Bug")

        if np.isnan(x).any():
            print("NaNs detected!")

        if self.sequence_length != DEFAULT_SEQUENCE_LEN:
            N, C, T = x.shape
            x = x.transpose(1, 0, 2).reshape(C, N * T)
            last_batch = False
            if not isinstance(self.sequence_length, str) and N * T != self.sequence_length * 128 * 60:
                last_batch = True
                x = np.pad(x, [(0, 0), (0, self.sequence_length * 128 * 60 - N * T)])
            N, C, T = t.shape
            t = t.transpose(1, 0, 2).reshape(C, N * T)
            if last_batch:
                t = np.pad(t, [(0, 0), (0, self.sequence_length * 60 - N * T)])
            current_sequence = np.arange(1000)[current_sequence]
            stable_sleep = stable_sleep.reshape(-1)
            if last_batch:
                stable_sleep = np.pad(stable_sleep, [(0, 2 * self.sequence_length - N * 10)])

        if isinstance(self.sequence_length, str) and self.sequence_length == "full":
            current_sequence = self.index_to_record[idx]["idx"]

        if scaler:
            try:
                x = scaler.transform(x.T).T  # (n_channels, n_samples)
            except:
                print("")

        ###############################################################################################
        # --------------------------- ADD SYNTHETIC SPINDLES ---------------------------------------- #

        # if current_record == "N4499_2 092211.h5" and current_sequence == 20:
        #     print("\nBAAAD BOIII")

        #     if True:
        #         # Add a synthetic spindle
        #         print("SPINDLEEEEEES")
        #         import matplotlib.pyplot as plt
        #         from scipy import signal
        #         import utils

        #         utils.plot_data(x.T, t.T, save_path="before_spindles.png")

        #         insertion_points = [
        #             (3 * 30 + 10) * 128,
        #             (3 * 30 + 2) * 128,
        #             (3 * 30 + 5) * 128,
        #             (3 * 30 + 10) * 128,
        #             (3 * 30 + 25) * 128,
        #         ]  # three 30 sepochs in plus 10 s, 128 Hz
        #         channels = [0, 1, 0, 1, 1]
        #         spindle_lengths = [5, 3, 2, 2, 3]  # seconds
        #         spindle_frequencies = 3 * np.random.sample(len(channels)) + 12  # Hz
        #         spindle_amplitude = 1.25

        #         augmented_x = np.zeros(x.T.shape)
        #         for idx, (insertion_point, spindle_length, channel, spindle_frequency) in enumerate(
        #             zip(insertion_points, spindle_lengths, channels, spindle_frequencies)
        #         ):
        #             spindle_t = np.linspace(
        #                 -spindle_length / 2, spindle_length / 2, spindle_length * 128, endpoint=False
        #             )
        #             spindle = spindle_amplitude * signal.gausspulse(
        #                 spindle_t, fc=spindle_frequency, bw=1 / (7.5 * spindle_length)
        #             )
        #             plt.figure()
        #             plt.plot(spindle_t, spindle)
        #             plt.xlabel("Time (s)")
        #             plt.savefig(f"results/synthetic_spindle_{idx:02}.png")

        #             augmented_x[insertion_point : insertion_point + spindle_length * 128, channel] = spindle

        #         x += augmented_x.T
        #         plt.figure(figsize=(16, 2))
        #         plt.plot(np.arange(0, 30 * 128) / 128, augmented_x[3 * 30 * 128 : 4 * 30 * 128, :2], linewidth=0.5)
        #         plt.xlabel("Time (s)")
        #         plt.ylim([-2, 2])
        #         plt.legend(["EEG C", "EEG O"])
        #         # plt.xlim([])
        #         plt.savefig("results/synthetic_spindles.png", dpi=300, bbox_inches="tight", pad_inches=0)
        #         utils.plot_data(x.T, t.T, save_path="more_spindles.png")
        assert x.shape[0] >= self.n_channels, f"Requested {self.n_channels}, PSG only has {x.shape[0]}!"
        if self.n_channels == 4:
            x = np.concatenate((x[np.newaxis, 0], x[-3:]), axis=0)  # only use 1 EEG
        elif self.n_channels == 5:
            pass  # H5 saves 5 channels per default. TODO: Change this when pipeline iis updated to include more channels, e.g. frontals.
        return x, t, current_record, current_sequence, stable_sleep

    def __str__(self):
        s = f"""
======================================
SSC-WSC PSG Dataset Dataset
--------------------------------------
Data directory: {self.data_dir}
Number of records: {len(self.records)}
======================================
"""

        return s


class SscWscPsgSubset(Dataset):
    def __init__(self, dataset, record_indices, name="Train", balanced_sampling=False):
        self.dataset = dataset
        self.record_indices = record_indices
        self.name = name
        self.balanced_sampling = balanced_sampling
        self.records = [self.dataset.records[idx] for idx in self.record_indices]
        self.sequence_indices = self.__get_subset_indices()

    def _get_subset_class_indices(self):
        records = set(self.records)
        out = {k: None for k in self.dataset.index_to_record_class.keys()}
        for c in out.keys():
            t = list(map(lambda x: x["record"] in records, self.dataset.index_to_record_class[c]))
            out[c] = list(compress(range(len(t)), t))
        return out

    def __get_subset_indices(self):
        records = set(self.records)
        t = list(map(lambda x: x["record"] in records, self.dataset.index_to_record))
        return list(compress(range(len(t)), t))

    def __getitem__(self, idx):
        if isinstance(self.sequence_indices, dict):
            class_choice = np.random.choice(list(self.sequence_indices.keys()))
            sequence_choice = np.random.choice(self.sequence_indices[class_choice])
            return self.dataset[self.dataset.index_to_record_class[class_choice][sequence_choice]["idx"]]
        else:
            return self.dataset[self.sequence_indices[idx]]

    def __len__(self):
        if isinstance(self.sequence_indices, dict):
            return sum([len(v) for v in self.sequence_indices.values()])
        else:
            return len(self.sequence_indices)

    def __str__(self):
        s = f"""
======================================
SSC-WSC PSG Dataset - {self.name} partition
--------------------------------------
Data directory: {self.dataset.data_dir}
Number of records: {len(self.record_indices)}
First ten records: {self.records[:10]}
======================================
"""

        return s


class SscWscDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        cv=None,
        cv_idx=None,
        data_dir=None,
        eval_ratio=0.1,
        n_workers=0,
        n_jobs=-1,
        n_records=None,
        scaling="robust",
        adjustment=None,
        balanced_sampling=False,
        sequence_length=5,
        max_eval_records=1000,
        **kwargs,
    ):
        super().__init__()
        self.adjustment = adjustment
        self.data_dir = data_dir
        self.balanced_sampling = balanced_sampling
        self.batch_size = batch_size
        self.cv = cv
        self.cv_idx = cv_idx
        self.eval_ratio = eval_ratio
        self.n_jobs = n_jobs
        self.n_records = n_records
        self.n_workers = n_workers
        self.scaling = scaling
        self.sequence_length = sequence_length
        self.max_eval_records = max_eval_records
        self.n_channels = kwargs["n_channels"]
        self.data = {"train": data_dir, "test": "data/test/raw"}
        self.dataset_params = dict(
            # data_dir=self.data_dir,
            cv=self.cv,
            cv_idx=self.cv_idx,
            eval_ratio=self.eval_ratio,
            n_jobs=self.n_jobs,
            n_records=self.n_records,
            scaling=self.scaling,
            adjustment=self.adjustment,
            sequence_length=self.sequence_length,
            max_eval_records=self.max_eval_records,
            n_channels=self.n_channels,
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            dataset = SscWscPsgDataset(
                data_dir=self.data["train"],
                balanced_sampling=self.balanced_sampling,
                overlap=self.balanced_sampling,
                **self.dataset_params,
            )
            self.train, self.eval = dataset.split_data()
        elif stage == "test":
            self.test = SscWscPsgDataset(data_dir=self.data["test"], overlap=False, **self.dataset_params)

    def train_dataloader(self):
        """Return training dataloader."""
        return torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return torch.utils.data.DataLoader(
            self.eval, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True, drop_last=True,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return torch.utils.data.DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True,
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        from argparse import ArgumentParser

        # DATASET specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        dataset_group = parser.add_argument_group("dataset")
        dataset_group.add_argument("--data_dir", default="data/train/raw/individual_encodings", type=str)
        dataset_group.add_argument("--eval_ratio", default=0.1, type=float)
        dataset_group.add_argument("--n_channels", default=5, type=int)
        dataset_group.add_argument("--n_jobs", default=-1, type=int)
        dataset_group.add_argument("--n_records", default=None, type=int)
        dataset_group.add_argument("--scaling", default=None, type=str)
        dataset_group.add_argument("--adjustment", default=0, type=int)
        dataset_group.add_argument("--cv", default=None, type=int)
        dataset_group.add_argument("--cv_idx", default=None, type=int)
        dataset_group.add_argument("--balanced_sampling", default=False, action="store_true")
        dataset_group.add_argument("--max_eval_records", default=500, type=int)
        dataset_group.add_argument(
            "--sequence_length",
            default=5,
            help="Sequence length in minutes (default: 5 min). If 'full', an entire PSG is loaded at a time (should only be used with batch_size=1).",
        )

        # DATALOADER specific
        dataloader_group = parser.add_argument_group("dataloader")
        dataloader_group.add_argument("--batch_size", default=1, type=int)
        dataloader_group.add_argument("--n_workers", default=0, type=int)

        return parser
