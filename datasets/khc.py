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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from tqdm import tqdm

from utils import ParallelExecutor
from utils import load_h5_data


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

    def __init__(self, data_dir=None, encoding="raw", n_jobs=-1, subset="test"):
        super().__init__()
        self.data_dir = data_dir
        self.encoding = encoding
        self.n_jobs = n_jobs
        self.subset = subset
        self.cohort = "khc"
        if self.data_dir is None:
            self.data_dir = os.path.join("data", self.subset, self.encoding, self.cohort)

        self.records = sorted(os.listdir(self.data_dir))
        self.index_to_record = []
        self.record_indices = {r: None for r in self.records}
        self.cache_dir = "data/.cache"
        memory = Memory(self.cache_dir, mmap_mode="r", verbose=0)
        get_data = memory.cache(load_psg_h5_data)

        # Get information about the data
        print(f"Loading mmap data using {n_jobs} workers:")
        data = ParallelExecutor(n_jobs=n_jobs, prefer="threads")(total=len(self.records))(
            delayed(get_data)(filename=os.path.join(self.data_dir, record)) for record in self.records
        )
        for record, sequences_in_file in zip(tqdm(self.records, desc="Processing"), data):
            self.record_indices[record] = np.arange(sequences_in_file)
            self.index_to_record.extend([{"record": record, "idx": x} for x in range(sequences_in_file)])
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

        return x, t, current_record, current_sequence

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
