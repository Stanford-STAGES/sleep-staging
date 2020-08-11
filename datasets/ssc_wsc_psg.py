import math
import os
import random

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

        return x, t

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
        self.records = [self.dataset.records[idx] for idx in self.record_indices]
        self.sequence_indices = [idx for idx, v in enumerate(self.dataset.index_to_record) for r in self.records if v['record'] == r]
        self.name = name

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


if __name__ == '__main__':

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
