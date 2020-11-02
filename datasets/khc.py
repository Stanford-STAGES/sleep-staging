import math
import os
import random
from itertools import compress

import numpy as np
import pandas as pd
import torch
from h5py import File
from joblib import delayed
from joblib import Memory
from joblib import Parallel
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from tqdm import tqdm

try:
    from utils import ParallelExecutor, load_h5_data
except ImportError:
    from utils.h5_utils import load_h5_data
    from utils.parallel_bar import ParallelExecutor

SCALERS = {"robust": RobustScaler, "standard": StandardScaler}


def get_unknown_stage(onehot_hypnogram):
    return onehot_hypnogram.sum(axis=1) == 0


def get_stable_stage(hypnogram, stage, adjustment=30):
    """
    Args:
        hypnogram (array_like): hypnogram with sleep stage labels
        stage (int): sleep stage label ({'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4})
        adjusted (int, optional): Controls the amount of bracketing surrounding a period of stable sleep.
        E.g. if adjustment=30, each period of stable sleep needs to be bracketed by 30 s.
    Returns:
        stable_periods: a list of range objects where each range describes a period of stable sleep stage.
    """
    from itertools import groupby
    from operator import itemgetter

    list_of_periods = []
    for k, g in groupby(enumerate(np.where(hypnogram == stage)[0]), lambda x: x[0] - x[1]):
        list_of_periods.append(list(map(itemgetter(1), g)))
    stable_periods = [range(period[0] + adjustment, period[-1] + 1 - adjustment) for period in list_of_periods]
    # Some periods are empty and need to be removed
    stable_periods = list(filter(lambda x: list(x), stable_periods))

    return stable_periods


def get_stable_sleep_periods(hypnogram, adjustment=30):
    """Get periods of stable sleep uninterrupted by transitions

    Args:
        hypnogram (array-like): hypnogram vector or array with sleep stage labels
        adjustment (int): parameter controlling the amount of shift when selecting periods of stable sleep. E.g.
        if adjustment = 30, each period of stable sleep needs to be bracketed by 30 s of the same sleep stage.
    """
    hypnogram_shape = hypnogram.shape
    hypnogram = hypnogram.reshape(np.prod(hypnogram_shape))
    stable_periods = []
    stable_periods_bool = np.full(np.prod(hypnogram_shape), False)
    for stage in [0, 1, 2, 3, 4]:
        stable_periods.append(get_stable_stage(hypnogram, stage, adjustment))
        for period in stable_periods[-1]:
            stable_periods_bool[period] = True
    stable_periods_bool = stable_periods_bool.reshape(hypnogram_shape)

    return stable_periods_bool, stable_periods


def initialize_record(filename, scaling=None, overlap=True, adjustment=30):

    if scaling in SCALERS.keys():
        scaler = SCALERS[scaling]()
    else:
        scaler = None

    with File(filename, "r") as h5:
        N, C, T = h5["M"].shape
        sequences_in_file = N

        if scaler:
            scaler.fit(h5["M"][:].transpose(1, 0, 2).reshape((C, N * T)).T)

        # Remember that the output array from the H5 has 50 % overlap between segments.
        # Use the following to split into even and odd
        if overlap:
            hyp_shape = h5["L"].shape
            hyp_even = h5["L"][0::2]
            hyp_odd = h5["L"][1::2]
            stable_sleep = np.full([v for idx, v in enumerate(hyp_shape) if idx != 1], False)
            stable_sleep[0::2] = get_stable_sleep_periods(hyp_even.argmax(axis=1), adjustment)[0]
            stable_sleep[1::2] = get_stable_sleep_periods(hyp_odd.argmax(axis=1), adjustment)[0]
        else:
            stable_sleep = get_stable_sleep_periods(h5["L"][:].argmax(axis=1), adjustment)[0]

        # Remove unknown stage
        unknown_stage = get_unknown_stage(h5["L"][:])
        stable_sleep[unknown_stage] = False

    return sequences_in_file, scaler, stable_sleep


def load_psg_h5_data(filename):

    with File(filename, "r") as h5:
        sequences_in_file = h5["M"].shape[0]
    return sequences_in_file
    # X = h5['M'][:].astype('float32')
    # y = h5['L'][:].astype('float32')

    # sequences_in_file = X.shape[0]

    # return X, y, sequences_in_file


class KoreanDataset(Dataset):
    """Korean Hypersomnia Cohort"""

    def __init__(self, data_dir=None, encoding="raw", n_jobs=-1, n_records=-1, subset="test", overlap=True, adjustment=30, scaling=None):
        super().__init__()
        self.data_dir = data_dir
        self.encoding = encoding
        self.n_jobs = n_jobs
        self.subset = subset
        self.cohort = "khc"
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
        self.cache_dir = "data/.cache"
        memory = Memory(self.cache_dir, mmap_mode="r", verbose=0)
        # get_data = memory.cache(load_psg_h5_data)
        get_data = memory.cache(initialize_record)

        # Get information about the data
        # print(f"Loading mmap data using {n_jobs} workers:")
        # data = ParallelExecutor(n_jobs=n_jobs, prefer="threads")(total=len(self.records))(
        #     delayed(get_data)(filename=os.path.join(self.data_dir, record)) for record in self.records
        # )
        # for record, sequences_in_file in zip(tqdm(self.records, desc="Processing"), data):
        #     self.record_indices[record] = np.arange(sequences_in_file)
        #     self.index_to_record.extend([{"record": record, "idx": x} for x in range(sequences_in_file)])
        # print("Finished loading data")

        # Get information about the data
        print(f"Loading mmap data using {n_jobs} workers:")
        data = ParallelExecutor(n_jobs=n_jobs, prefer="threads")(total=len(self.records))(
            delayed(get_data)(
                filename=os.path.join(self.data_dir, record), scaling=self.scaling, adjustment=self.adjustment, overlap=self.overlap
            )
            for record in self.records
        )
        # for record, d in zip(tqdm(self.records, desc='Processing'), data):
        #     seqs_in_file = d[2]
        #     self.data[record] = {'data': d[0], 'target': d[1]}
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
                t = f["L"][current_sequence].astype("uint8")
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
Korean Hypersomnia Cohort Dataset
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


def collate_fn(batch):

    X, y = map(torch.FloatTensor, zip(*batch))

    return X, y
    # return torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(w)


# class KoreanSubset(Dataset):
#     def __init__(self, dataset, record_indices, name="Train"):
#         self.dataset = dataset
#         self.record_indices = record_indices
#         self.name = name
#         self.records = [self.dataset.records[idx] for idx in self.record_indices]
#         self.sequence_indices = (
#             self.__get_subset_indices()
#         )  # [idx for idx, v in enumerate(self.dataset.index_to_record) for r in self.records if v['record'] == r]# [idx for idx, v in enumerate(self.dataset.index_to_record) for r in self.records if v['record'] == r]

#     def __get_subset_indices(self):
#         t = list(map(lambda x: x["record"] in self.records, self.dataset.index_to_record))
#         return list(compress(range(len(t)), t))

#     def __getitem__(self, idx):
#         return self.dataset[self.sequence_indices[idx]]

#     def __len__(self):
#         return len(self.sequence_indices)

#     def __str__(self):
#         s = f"""
# ======================================
# Korean Hypersomnia Cohort Dataset - {self.name} partition
# --------------------------------------
# Data directory: {self.dataset.data_dir}
# Number of records: {len(self.record_indices)}
# First ten records: {self.records[:10]}
# ======================================
# """

#         return s


if __name__ == "__main__":

    from tqdm import tqdm

    np.random.seed(42)
    random.seed(42)

    # dataset_params = dict(data_dir="./data/raw/individual_encodings", n_jobs=-1)
    dataset_params = {}
    dataset = KoreanDataset(**dataset_params)
    print(dataset)
    dataloader_params = dict(batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    pbar = tqdm(DataLoader(dataset, **dataloader_params))
    for idx, (x, t, record, seq_nr) in enumerate(pbar):
        if idx == 0:
            print(f"{record}: {x.shape}")
