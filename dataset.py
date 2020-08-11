import math
import os
import random

import numpy as np
import torch
from joblib import delayed
from joblib import Memory
from joblib import Parallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from tqdm import tqdm

from utils import ParallelExecutor, load_h5_data


class StagesData(Dataset):

    def __init__(self, data_dir=None, batch_size=15, n_classes=5, n_jobs=-1, num_steps=None, seg_size=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_jobs = n_jobs
        self.n_steps = num_steps
        self.seg_size = seg_size

        self.records = sorted(os.listdir(self.data_dir))[:-1]
        self.data = {r: [] for r in self.records}
        self.index_to_record = []
        self.record_to_index = []
        self.record_indices = {r: None for r in self.records}
        self.batch_indices = []
        self.current_record_idx = -1
        self.current_record = None
        self.loaded_record = None
        self.current_position = None
        print(self.record_to_index)
        # data = load_h5_data(os.path.join(self.data_dir, self.records[0]))
        self.cache_dir = 'data/.cache'
        memory = Memory(self.cache_dir, mmap_mode='r', verbose=0)
        get_data = memory.cache(load_h5_data)

        print(f'Loading mmap data using {n_jobs} workers:')
        data = ParallelExecutor(n_jobs=n_jobs, prefer="threads")(total=len(self.records))(
            delayed(get_data)(filename=os.path.join(self.data_dir, record), seg_size=self.seg_size) for record in self.records
        )
        # print('Processing...')
        self.index_matrix = []
        for record, d in zip(tqdm(self.records, desc='Processing'), data):
            seqs_in_file = d[3]
            self.data[record] = {'data': d[0], 'target': d[1], 'weights': d[2]}
            self.record_indices[record] = np.arange(seqs_in_file)
            self.index_to_record.extend(
                [
                    {'record': record, 'idx': x} for x in range(seqs_in_file)
                ]
            )
            self.batch_indices.extend(
                [
                    {'record': record, 'range': np.arange(v, v + self.batch_size)} for v in range(0, seqs_in_file, self.batch_size)
                ]
            )
            self.record_to_index.append(
                {'record': record, 'range': np.arange(seqs_in_file)})
            # Define a matrix of indices
            self.index_matrix.extend(
                [np.arange(seqs_in_file)]
            )
            # print(record)
        self.index_matrix = np.stack(self.index_matrix)
        print('Finished loading data')

        # # Define a matrix of indices
        # self.index_matrix = np.stack(
        #     [np.arange(0, 300) for _ in range(len(self.records))])
        print(self.index_matrix.shape)
        self.shuffle_records()
        self.shuffle_data()
        self.batch_data()
        # self.initialize()

        # # Preload first h5 into memory
        # self.current_record = self.index_to_record[0]['record']
        # self.loaded_record = None
        # self.get_record()

    def initialize(self):
        self.current_record_idx = -1
        self.set_next()

    def batch_data(self):
        self.batch_indices = []
        self.batch_indices.extend([
            {'range': split, 'record': k} for k, v in self.record_indices.items() for split in np.split(v, 300 / self.batch_size)
        ])

    def shuffle_records(self):
        random.shuffle(self.records)

    def shuffle_data(self):

        # Shuffle subjects
        random.shuffle(self.records)

        # Shuffle each record
        # [random.shuffle(v['range']) for v in self.record_to_index]
        [random.shuffle(v) for v in self.record_indices.values()]

        # # First shuffle each row independently
        # print(self.index_matrix)
        # x, y = self.index_matrix.shape
        # rows = np.indices((x, y))[0]
        # cols = [np.random.permutation(y) for _ in range(x)]
        # self.index_matrix = self.index_matrix[rows, cols]
        # # np.random.shuffle(self.index_matrix.T)
        # print(self.index_matrix)
        # # Randomly assign row numbers to records
        # self.record_to_index = {r: v for r, v in zip(
        #     records, np.random.permutation(np.arange(len(self.records))))}
        # print(self.record_to_index)

    def split_data(self, ratio):
        n_records = len(self.records)
        n_eval = int(n_records * ratio)
        n_train = n_records - n_eval
        self.shuffle_data()
        self.batch_data()
        train_data = Subset(self, np.arange(n_eval, n_records))
        eval_data = Subset(self, np.arange(0, n_eval))
        return train_data, eval_data

    def get_record(self):
        # if not self.loaded_record:
        self.loaded_record = {'data': self.data[self.current_record]['data'],
                              'target': self.data[self.current_record]['target'],
                              'weights': self.data[self.current_record]['weights'],
                              'record': self.current_record,
                              'current_pos': 0,
                              }
        # }
        # elif self.loaded_record['record'] != self.current_record:
        # self.
        # self.loaded_record = {k: v for k, v in zip(['data', 'target', 'weights'], load_h5_data(
        #     os.path.join(self.data_dir, self.current_record), seg_size=self.seg_size))}
        # self.loaded_record.update(
        #     {'record': self.current_record, 'current_pos': 0})
        # print('debug')

    def set_next(self):
        self.current_record_idx += 1
        self.current_record = self.records[self.current_record_idx]
        self.current_position = 0
        # self.get_record()

    def __len__(self):
        # return 66
        # return len(self.index_to_record)
        # return len(self.index_matrix.flatten())
        # return sum([len(v['range']) for v in self.record_to_index])
        return math.ceil(sum([len(v) for v in self.record_indices.values()]) / self.batch_size)

    def __getitem__(self, idx):

        try:
            # if not self.current_record:
            #     self.set_next()
            # # record = self.records[idx]
            # if self.current_position >= len(self.record_indices[self.current_record]):
            # # if self.loaded_record['current_pos'] >= len(self.record_indices[self.current_record]):
            #     self.set_next()

            # Grab data
            current_record = self.batch_indices[idx]['record']
            samples = self.batch_indices[idx]['range']

            # Grab data
            # current_record = self.current_record
            # current_pos = self.current_position
            # samples = self.record_indices[current_record][current_pos:current_pos+self.batch_size]
            x = self.data[current_record]['data'][samples]
            t = self.data[current_record]['target'][samples]
            # x = self.loaded_record['data'][samples]
            w = self.data[current_record]['weights'][samples]
            # t = self.loaded_record['target'][samples]
            # w = self.loaded_record['weights'][samples]

            # Update current position
            # self.current_position += len(samples)
            # self.loaded_record['current_pos'] += len(samples)
            # record = self.index_to_record[idx]['record']
            # record_idx = self.index_to_record[idx]['idx']
            # index = self.record_to_index[record]
            # if self.loaded_record['record'] != record:
            #     self.current_record = record
            #     self.get_record()

            # while self.loaded_record['current_pos'] < 300:
            #    x = self.loaded_record['data'][self.loaded_record['current_pos'], :, :]
            #    t = self.loaded_record['target'][self.loaded_record['current_pos'], :, :]
            #    w = self.loaded_record['weights'][self.loaded_record['current_pos'], :]
            #    self.loaded_record['current_pos'] += 1
            #    return x, t, w
            # x = self.loaded_record['data'][]
            # x = self.data[record]['data'][record_idx, :, :]
            # t = self.data[record]['target'][record_idx, :, :]
            # w = self.data[record]['weights'][record_idx, :]
        except IndexError:
            print('Bug')

        if np.isnan(x).any():
            print('NaNs detected!')
            x[np.isnan(x)] = 2 ** -23
            x[np.isinf(x)] = 2 ** -23

        return x, t, w
        # return (np.reshape(x, [n_segs, self.seg_size, n_features]),
        #         np.reshape(t, [n_segs, self.seg_size, self.n_classes]),
        #         np.reshape(w, [n_segs, self.seg_size, 1]))

    def __str__(self):
        s = f"""
======================================
STAGES Dataset
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

    X, y, w = map(torch.FloatTensor, zip(*batch))

    return X.squeeze(), y.squeeze(), w.squeeze()
    # return torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(w)


class Subset(Dataset):
    def __init__(self, dataset, record_indices, name='Train'):
        self.dataset = dataset
        self.record_indices = record_indices
        self.records = [self.dataset.records[idx] for idx in self.record_indices]
        self.batch_indices = [idx for idx, v in enumerate(
            self.dataset.batch_indices) for r in self.records if v['record'] == r]
        self.name = name

    def __getitem__(self, idx):
        return self.dataset[self.batch_indices[idx]]

    def __len__(self):
        return len(self.batch_indices)

    def __str__(self):
        s = f"""
======================================
STAGES Dataset - {self.name} partition
--------------------------------------
Data directory: {self.dataset.data_dir}
Number of records: {len(self.record_indices)}
Records: {self.records}
======================================
"""

        return s


if __name__ == '__main__':

    from tqdm import tqdm

    dataset_params = dict(data_dir='./data/batch_encodings', batch_size=30,
                          num_steps=50000, n_jobs=1, n_classes=5, seg_size=60)
    dataset = StagesData(**dataset_params)
    train_data, eval_data = dataset.split_data(0.1)
    print(train_data)
    pbar = tqdm(DataLoader(train_data, batch_size=None, shuffle=False, num_workers=0, pin_memory=True))
    for x, t, w in pbar:
        pass
    print(len(dataset))
    print(dataset)
    next(iter(dataset))
    dataset.initialize()
    train_data, eval_data = dataset.split_data(0.10)
    print(train_data)
    print(eval_data)
    train_loader = DataLoader(train_data, batch_size=None, num_workers=5, shuffle=False)
    eval_loader = DataLoader(eval_data, batch_size=None, num_workers=5, shuffle=False)
    for b in tqdm(train_loader):
        pass
    for b in tqdm(eval_loader):
        pass
    # bar = tqdm(range(len(dataset)))
    # for idx in tqdm(range(len(dataset))):
    #    b = dataset[idx]
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False)
    # dataloader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn, num_workers=0, shuffle=False)
    # dataloader = DataLoader(dataset, batch_sampler=torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(range(300)), batch_size=10, drop_last=True), collate_fn=collate_fn, num_workers=0, shuffle=False)

    bar = tqdm(dataloader, leave=False)
    for b in bar:
        pass
        # print(b[0].shape)
        # print(f'x.shape: {b[0].shape}')
    # batch = next(iter(dataloader))
    # print(batch)
