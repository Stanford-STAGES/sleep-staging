import math
import os
import random
from itertools import compress

import numpy as np
import torch
from h5py import File
from joblib import delayed
from joblib import Memory
from joblib import Parallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import ParallelExecutor, load_h5_data, read_fns


# def load_psg_h5_data(filename, scaling=None):
#     scaler = None

#     if scaling:
#         scaler = SCALERS[scaling]()

#     with File(filename, "r") as h5:
#         N, C, T = h5["M"].shape
#         sequences_in_file = N

#         if scaling:
#             scaler.fit(h5["M"][:].transpose(1, 0, 2).reshape((C, N * T)).T)

#     return sequences_in_file, scaler


class BaseCohortDataset(Dataset):
    def __init__(self, data_dir=None, n_jobs=None, scaling=None, n_records=None, psg_read_fn=None, cohort=None):
        assert data_dir and n_jobs and scaling and n_records and psg_read_fn, "Please supply input arguments!"
        super().__init__()
        self.data_dir = data_dir
        self.n_jobs = n_jobs
        self.scaling = scaling
        self.n_records = n_records
        self.psg_read_fn = psg_read_fn
        self.cohort = cohort

        self.records = sorted(os.listdir(self.data_dir))[: self.n_records]
        self.index_to_record = []
        self.record_indices = {r: None for r in self.records}
        self.scalers = {r: None for r in self.records}
        self.cache_dir = "data/.cache"
        memory = Memory(self.cache_dir, mmap_mode="r", verbose=0)
        get_data = memory.cache(self.psg_read_fn)

        # Get information about the data
        print(f"Loading mmap data using {n_jobs} workers:")
        data = ParallelExecutor(n_jobs=n_jobs, prefer="threads")(total=len(self.records))(
            delayed(get_data)(filename=os.path.join(self.data_dir, record), scaling=self.scaling) for record in self.records
        )
        for record, (sequences_in_file, scaler) in zip(tqdm(self.records, desc="Processing"), data):
            self.record_indices[record] = np.arange(sequences_in_file)
            self.index_to_record.extend([{"record": record, "idx": x} for x in range(sequences_in_file)])
            self.scalers[record] = scaler
        print("Finished loading data")

    def shuffle_records(self):
        random.shuffle(self.records)

    def split_data(self, ratio):
        n_records = len(self.records)
        n_eval = int(n_records * ratio)
        n_train = n_records - n_eval
        self.shuffle_records()
        train_data = Subset(self, np.arange(n_eval, n_records), cohort=self.cohort, subset="train")
        eval_data = Subset(self, np.arange(0, n_eval), cohort=self.cohort, subset="eval")
        return train_data, eval_data

    def __len__(self):
        return len(self.index_to_record)

    def __getitem__(self, idx):

        try:
            # Grab data
            current_record = self.index_to_record[idx]["record"]
            current_sequence = self.index_to_record[idx]["idx"]
            scaler = self.scalers[current_record]

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

        return x, t, current_record, current_sequence

    def __str__(self):
        s = f"""
======================================
PSG Dataset
--------------------------------------
Cohort: {self.cohort['name']}
Encoding: {self.cohort['encoding']}
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


class Subset(Dataset):
    def __init__(self, dataset, record_indices, cohort=None, subset="Train"):
        self.dataset = dataset
        self.record_indices = record_indices
        self.cohort = cohort
        self.subset = subset
        self.records = [self.dataset.records[idx] for idx in self.record_indices]
        self.sequence_indices = (
            self.__get_subset_indices()
        )  # [idx for idx, v in enumerate(self.dataset.index_to_record) for r in self.records if v['record'] == r]# [idx for idx, v in enumerate(self.dataset.index_to_record) for r in self.records if v['record'] == r]

    def __get_subset_indices(self):
        t = list(map(lambda x: x["record"] in self.records, self.dataset.index_to_record))
        return list(compress(range(len(t)), t))

    def __getitem__(self, idx):
        return self.dataset[self.sequence_indices[idx]]

    def __len__(self):
        return len(self.sequence_indices)

    def __str__(self):
        s = f"""
======================================
PSG Dataset
--------------------------------------
Cohort: {self.cohort['name']}
Encoding: {self.cohort['encoding']}
Subset: {self.subset}
Data directory: {self.dataset.data_dir}
Number of records: {len(self.record_indices)}
First ten records: {self.records[:10]}
======================================
"""

        return s


if __name__ == "__main__":

    from tqdm import tqdm

    np.random.seed(42)
    random.seed(42)

    # dataset_params = dict(data_dir="./data/raw/individual_encodings", n_jobs=-1, scaling="robust", n_records=10)
    # dataset = SscWscPsgDataset(**dataset_params)
    dataset_params = dict(
        data_dir="data/ssc/train/raw",
        n_jobs=-1,
        scaling="robust",
        n_records=10,
        psg_read_fn=read_fns["raw"],
        cohort={"name": "SSC", "encoding": "raw"},
    )
    dataset = BaseCohortDataset(**dataset_params)
    print(dataset)
    train_data, eval_data = dataset.split_data(0.1)
    print(train_data)
    pbar = tqdm(DataLoader(train_data, batch_size=57, shuffle=True, num_workers=0, pin_memory=True))
    for idx, (x, t, record, seq_nr) in enumerate(pbar):
        pass
    print(f"Record: {record}")
    print(f"Sequence nr: {seq_nr}")
    print(f"x.shape: {x.shape}")
    print(eval_data)
    pbar = tqdm(DataLoader(eval_data, batch_size=57, shuffle=True, num_workers=20, pin_memory=True))
    for idx, (x, t, record, seq_nr) in enumerate(pbar):
        pass
    print(f"Record: {record}")
    print(f"Sequence nr: {seq_nr}")
    print(f"x.shape: {x.shape}")
