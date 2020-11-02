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
from torch.utils.data import Subset
from tqdm import tqdm

from utils import ParallelExecutor, load_h5_data


def load_psg_h5_data(filename):

    with File(filename, 'r') as h5:
        sequences_in_file = h5['M'].shape[0]
    return sequences_in_file
        # X = h5['M'][:].astype('float32')
        # y = h5['L'][:].astype('float32')

    # sequences_in_file = X.shape[0]

    # return X, y, sequences_in_file


class SscWscPsgDataset(Dataset):

    def __init__(self, data_dir=None, n_jobs=-1):
        super().__init__()
        self.data_dir = data_dir
        self.n_jobs = n_jobs
        self.batch_size = 32

        self.records = sorted(os.listdir(self.data_dir))
        # self.data = {r: [] for r in self.records}
        self.index_to_record = []
        # self.record_to_index = []
        self.record_indices = {r: None for r in self.records}
        # self.batch_indices = []
        # self.current_record_idx = -1
        # self.current_record = None
        # self.loaded_record = None
        # self.current_position = None
        # data = load_psg_h5_data(os.path.join(self.data_dir, self.records[0]))
        self.cache_dir = 'data/.cache'
        memory = Memory(self.cache_dir, mmap_mode='r', verbose=0)
        get_data = memory.cache(load_psg_h5_data)

        # Get information about the data
        print(f'Loading mmap data using {n_jobs} workers:')
        data = ParallelExecutor(n_jobs=n_jobs, prefer="threads")(total=len(self.records))(
            delayed(get_data)(filename=os.path.join(self.data_dir, record)) for record in self.records
        )
        # for record, d in zip(tqdm(self.records, desc='Processing'), data):
        #     seqs_in_file = d[2]
        #     self.data[record] = {'data': d[0], 'target': d[1]}
        for record, sequences_in_file in zip(tqdm(self.records, desc='Processing'), data):
            self.record_indices[record] = np.arange(sequences_in_file)
            self.index_to_record.extend([
                {'record': record, 'idx': x} for x in range(sequences_in_file)
            ])
        print('Finished loading data')

    def shuffle_records(self):
        random.shuffle(self.records)

    def split_data(self, ratio):
        n_records = len(self.records)
        n_eval = int(n_records * ratio)
        n_train = n_records - n_eval
        self.shuffle_records()
        train_data = SscWscPsgSubset(self, np.arange(n_eval, n_records), name='Train')
        eval_data = SscWscPsgSubset(self, np.arange(0, n_eval), name='Validation')
        return train_data, eval_data

    def __len__(self):
        return len(self.index_to_record)

    def __getitem__(self, idx):

        try:
            # Grab data
            current_record = self.index_to_record[idx]['record']
            current_sequence = self.index_to_record[idx]['idx']

            # Grab data
            with File(os.path.join(self.data_dir, current_record), 'r') as f:
                x = f['M'][current_sequence].astype('float32')
                t = f['L'][current_sequence].astype('uint8')
            # x = self.data[current_record]['data'][current_sequence]
            # t = self.data[current_record]['target'][current_sequence]

        except IndexError:
            print('Bug')

        if np.isnan(x).any():
            print('NaNs detected!')

        return x, t, current_record, current_sequence

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


class SscWscPsgSubset(Dataset):
    def __init__(self, dataset, record_indices, name='Train'):
        self.dataset = dataset
        self.record_indices = record_indices
        self.name = name
        self.records = [self.dataset.records[idx] for idx in self.record_indices]
        self.sequence_indices = self.__get_subset_indices()# [idx for idx, v in enumerate(self.dataset.index_to_record) for r in self.records if v['record'] == r]# [idx for idx, v in enumerate(self.dataset.index_to_record) for r in self.records if v['record'] == r]

    def __get_subset_indices(self):
        t = list(map(lambda x: x['record'] in self.records, self.dataset.index_to_record))
        return list(compress(range(len(t)), t))

    def __getitem__(self, idx):
        return self.dataset[self.sequence_indices[idx]]

    def __len__(self):
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
        n_records=-1,
        scaling="robust",
        adjustment=None,
        **kwargs,
    ):
        super().__init__()
        self.adjustment = adjustment
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.cv = cv
        self.cv_idx = cv_idx
        self.eval_ratio = eval_ratio
        self.n_jobs = n_jobs
        self.n_records = n_records
        self.n_workers = n_workers
        self.scaling = scaling
        self.data = {"train": os.path.join(data_dir, "train"), "test": os.path.join(data_dir, "test")}
        self.dataset_params = dict(
            # data_dir=self.data_dir,
            cv=self.cv,
            cv_idx=self.cv_idx,
            eval_ratio=self.eval_ratio,
            n_jobs=self.n_jobs,
            n_records=self.n_records,
            scaling=self.scaling,
            adjustment=self.adjustment,
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            dataset = SscWscPsgDataset(data_dir=self.data["train"], **self.dataset_params)
            self.train, self.eval = dataset.split_data()
        elif stage == "test":
            self.test = SscWscPsgDataset(data_dir=self.data["test"], overlap=False, **self.dataset_params)

    def train_dataloader(self):
        """Return training dataloader."""
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return torch.utils.data.DataLoader(
            self.eval,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
            drop_last=True,
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
        dataset_group.add_argument("--n_jobs", default=-1, type=int)
        dataset_group.add_argument("--n_records", default=-1, type=int)
        dataset_group.add_argument("--scaling", default=None, type=str)
        dataset_group.add_argument("--adjustment", default=0, type=int)
        dataset_group.add_argument("--cv", default=None, type=int)
        dataset_group.add_argument("--cv_idx", default=None, type=int)

        # DATALOADER specific
        dataloader_group = parser.add_argument_group("dataloader")
        dataloader_group.add_argument("--batch_size", default=64, type=int)
        dataloader_group.add_argument("--n_workers", default=20, type=int)

        return parser


if __name__ == "__main__":

    from tqdm import tqdm
    np.random.seed(42)
    random.seed(42)

    dataset_params = dict(data_dir='./data/raw/individual_encodings', n_jobs=-1)
    dataset = SscWscPsgDataset(**dataset_params)
    print(dataset)
    train_data, eval_data = dataset.split_data(0.1)
    print(train_data)
    pbar = tqdm(DataLoader(train_data, batch_size=32, shuffle=True, num_workers=20, pin_memory=True))
    for idx, (x, t) in enumerate(pbar):
        if idx == 0:
            print(x.shape)
    print(eval_data)
    pbar = tqdm(DataLoader(eval_data, batch_size=32, shuffle=True, num_workers=20, pin_memory=True))
    for x, t in pbar:
        pass
