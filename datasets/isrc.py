import argparse
import math
import os
import random
from itertools import compress

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from h5py import File
from joblib import delayed
from joblib import Memory
from joblib import Parallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from tqdm import tqdm

import utils


# def load_psg_h5_data(filename):

#     with File(filename, "r") as h5:
#         sequences_in_file = h5["M"].shape[0]
#     return sequences_in_file
# X = h5['M'][:].astype('float32')
# y = h5['L'][:].astype('float32')

# sequences_in_file = X.shape[0]

# return X, y, sequences_in_file


class ISRCDataset(Dataset):
    """Interscorer Reliability Cohort Dataset"""

    def __init__(
        self,
        data_dir=None,
        encoding="raw",
        n_jobs=-1,
        n_records=-1,
        subset="test",
        overlap=True,
        adjustment=30,
        scaling=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.encoding = encoding
        self.n_jobs = n_jobs
        self.subset = subset
        self.cohort = "isrc"
        self.scaling = scaling
        self.adjustment = adjustment
        self.n_records = n_records
        self.overlap = overlap
        if self.data_dir is None:
            self.data_dir = os.path.join("data", self.subset, self.encoding, self.cohort)

        self.records = sorted(os.listdir(self.data_dir))[: self.n_records]
        self.index_to_record = []
        self.record_indices = {r: None for r in self.records}
        self.scalers = {r: None for r in self.records}
        self.stable_sleep = {r: None for r in self.records}
        self.cache_dir = ".cache"
        memory = Memory(self.cache_dir, mmap_mode="r", verbose=0)
        get_data = memory.cache(utils.initialize_record)

        # Get information about the data
        print(f"Loading mmap data using {self.n_jobs} workers:")
        data = utils.ParallelExecutor(n_jobs=self.n_jobs, prefer="threads")(total=len(self.records))(
            delayed(get_data)(
                filename=os.path.join(self.data_dir, record),
                scaling=self.scaling,
                adjustment=self.adjustment,
                overlap=self.overlap,
            )
            for record in self.records
        )
        for record, (sequences_in_file, scaler, stable_sleep) in zip(tqdm(self.records, desc="Processing"), data):
            self.record_indices[record] = np.arange(sequences_in_file)
            self.index_to_record.extend([{"record": record, "idx": x} for x in range(sequences_in_file)])
            self.scalers[record] = scaler
            self.stable_sleep[record] = stable_sleep
        print("Finished loading data")

    # def shuffle_records(self):
    #     random.shuffle(self.records)

    # def split_data(self, ratio):
    #     n_records = len(self.records)
    #     n_eval = int(n_records * ratio)
    #     n_train = n_records - n_eval
    #     self.shuffle_records()
    #     train_data = SscWscPsgSubset(self, np.arange(n_eval, n_records), name="Train")
    #     eval_data = SscWscPsgSubset(self, np.arange(0, n_eval), name="Validation")
    #     return train_data, eval_data

    def __len__(self):
        return len(self.index_to_record)

    def __getitem__(self, idx):

        try:
            # Grab data
            current_record = self.index_to_record[idx]["record"]
            current_sequence = self.index_to_record[idx]["idx"]
            scaler = self.scalers[current_record]
            stable_sleep = np.array(self.stable_sleep[current_record][current_sequence]).squeeze()

            # Grab data
            with File(os.path.join(self.data_dir, current_record), "r") as f:
                x = f["M"][current_sequence].astype("float32")
                t = f["L"][current_sequence].astype("uint8").squeeze()
            # x = self.data[current_record]['data'][current_sequence]
            # t = self.data[current_record]['target'][current_sequence]

        except IndexError:
            print("Bug")

        if np.isnan(x).any():
            print("NaNs detected!")

        if scaler:
            x = scaler.transform(x.T).T  # (n_channels, n_samples)

        return x, t, current_record, current_sequence, stable_sleep

    def __str__(self):
        s = f"""
======================================
Interscorer Reliability Cohort Dataset
--------------------------------------
Data directory: {self.data_dir}
Number of records: {len(self.records)}
======================================
"""
        return s


# def collate_fn(batch):
#
#    x, t, w = (
#        np.stack([b[0] for b in batch]),
#        np.stack([b[1] for b in batch]),
#        np.stack([b[2] for b in batch])
#    )
#
#    return torch.FloatTensor(x), torch.IntTensor(t), torch.FloatTensor(w)


# def collate_fn(batch):

#     X, y = map(torch.FloatTensor, zip(*batch))

#     return X, y
# return torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(w)


# class SscWscPsgSubset(Dataset):
#     def __init__(self, dataset, record_indices, name="Train"):
#         self.dataset = dataset
#         self.record_indices = record_indices
#         self.records = [self.dataset.records[idx] for idx in self.record_indices]
#         self.sequence_indices = [
#             idx for idx, v in enumerate(self.dataset.index_to_record) for r in self.records if v["record"] == r
#         ]
#         self.name = name

#     def __getitem__(self, idx):
#         return self.dataset[self.sequence_indices[idx]]

#     def __len__(self):
#         return len(self.sequence_indices)

#     def __str__(self):
#         s = f"""
# ======================================
# SSC-WSC PSG Dataset - {self.name} partition
# --------------------------------------
# Data directory: {self.dataset.data_dir}
# Number of records: {len(self.record_indices)}
# First ten records: {self.records[:10]}
# ======================================
# """

#         return s


class ISRCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        data_dir=None,
        n_workers=0,
        n_jobs=-1,
        n_records=None,
        scaling="robust",
        adjustment=None,
        **kwargs,
    ):
        super().__init__()
        self.adjustment = adjustment
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.n_records = n_records
        self.n_workers = n_workers
        self.scaling = scaling
        self.dataset_params = dict(
            # data_dir=self.data_dir,
            n_jobs=self.n_jobs,
            n_records=self.n_records,
            scaling=self.scaling,
            adjustment=self.adjustment,
        )

    def setup(self, stage="test"):
        if stage == "fit":
            raise NotImplementedError
        elif stage == "test":
            self.test = ISRCDataset(data_dir=self.data_dir, overlap=False, **self.dataset_params)

    def test_dataloader(self):
        """Return test dataloader."""
        return torch.utils.data.DataLoader(
            self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True,
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser):

        # DATASET specific
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        dataset_group = parser.add_argument_group("dataset")
        dataset_group.add_argument("--data_dir", default="data/isrc/raw/5min", type=str)
        dataset_group.add_argument("--n_jobs", default=-1, type=int)
        dataset_group.add_argument("--n_records", default=None, type=int)
        dataset_group.add_argument("--scaling", default="robust", type=str)
        dataset_group.add_argument("--adjustment", default=0, type=int)

        # DATALOADER specific
        dataloader_group = parser.add_argument_group("dataloader")
        dataloader_group.add_argument("--batch_size", default=64, type=int)
        dataloader_group.add_argument("--n_workers", default=20, type=int)

        return parser


if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser = ISRCDataModule.add_dataset_specific_args(parser)
    args = parser.parse_args()

    # args.n_workers = 10

    print(args)

    isrc = ISRCDataModule(**vars(args))
    isrc.setup("test")
    isrc_test = isrc.test_dataloader()

    for idx, batch in enumerate(tqdm(isrc_test)):
        if idx == 0:
            print(batch)

    # dataset_params = dict(data_dir="./data/raw/individual_encodings", n_jobs=-1)
    # dataset_params = {}
    # dataset = ISRCDataset(**dataset_params)
    # print(dataset)
    # dataloader_params = dict(batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    # pbar = tqdm(DataLoader(dataset, **dataloader_params))
    # for idx, (x, t, record, seq_nr) in enumerate(pbar):
    #     if idx == 0:
    #         print(f"Record: {record}\nX.shape: {x.shape}\nt.shape: {t.shape}")
