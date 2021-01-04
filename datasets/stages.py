import math
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from h5py import File
from joblib import delayed
from joblib import Memory
from tqdm import tqdm

try:
    from utils import load_h5_data
    from utils import get_h5_info
    from utils import ParallelExecutor
except ImportError:
    from utils.h5_utils import load_h5_data
    from utils.parallel_bar import ParallelExecutor


class STAGESDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir=None,
        batch_size=None,
        eval_ratio=None,
        n_classes=None,
        n_jobs=None,
        n_records=None,
        seg_size=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.eval_ratio = eval_ratio
        self.n_classes = n_classes
        self.n_jobs = n_jobs
        self.n_records = n_records
        self.seg_size = seg_size

        self.records = sorted(os.listdir(self.data_dir))[: self.n_records]
        self.data = {r: [] for r in self.records}
        # self.index_to_record = []
        # self.record_to_index = []
        self.record_indices = {r: None for r in self.records}
        self.batch_indices = []
        # self.current_record_idx = -1
        # self.current_record = None
        # self.loaded_record = None
        # self.current_position = None
        # print(self.record_to_index)
        # data = load_h5_data(os.path.join(self.data_dir, self.records[0]))
        self.cache_dir = "data/.cache"
        memory = Memory(self.cache_dir, mmap_mode="r", verbose=0)
        get_data = memory.cache(get_h5_info)
        # get_data = memory.cache(load_h5_data)

        print(f"Loading mmap data using {n_jobs} workers:")
        # data = ParallelExecutor(n_jobs=n_jobs, prefer="threads")(total=len(self.records))(
        #     delayed(get_data)(filename=os.path.join(self.data_dir, record), seg_size=self.seg_size)
        #     for record in self.records
        # )
        data = ParallelExecutor(n_jobs=n_jobs, prefer="threads")(total=len(self.records))(
            delayed(get_data)(filename=os.path.join(self.data_dir, record)) for record in self.records
        )
        # print('Processing...')
        # self.index_matrix = []
        for d, record in tqdm(data, desc="Processing"):
            # for record, d in zip(tqdm(self.records, desc="Processing"), data):
            # seqs_in_file = d[3]
            # self.data[record] = {"data": d[0], "target": d[1], "weights": d[2]}
            seqs_in_file = d
            self.record_indices[record] = np.arange(seqs_in_file)
            # self.index_to_record.extend([{"record": record, "idx": x} for x in range(seqs_in_file)])
            # self.batch_indices.extend(
            #     [
            #         {"record": record, "range": np.arange(v, v + self.batch_size)}
            #         for v in range(0, seqs_in_file, self.batch_size)
            #     ]
            # )
            # self.record_to_index.append({"record": record, "range": np.arange(seqs_in_file)})
            # Define a matrix of indices
            # self.index_matrix.extend([np.arange(seqs_in_file)])
            # print(record)
        # self.index_matrix = np.stack(self.index_matrix)
        print("Finished loading data")

        # # Define a matrix of indices
        # self.index_matrix = np.stack(
        #     [np.arange(0, 300) for _ in range(len(self.records))])
        # print(self.index_matrix.shape)
        # self.shuffle_records()
        self.shuffle_data()
        self.batch_data()
        # self.initialize()

        # # Preload first h5 into memory
        # self.current_record = self.index_to_record[0]['record']
        # self.loaded_record = None
        # self.get_record()

    def shuffle_data(self):

        # Shuffle subjects
        self.shuffle_records()

        # Shuffle each record
        [random.shuffle(v) for v in self.record_indices.values()]

    def shuffle_records(self):
        random.shuffle(self.records)

    def batch_data(self):
        self.batch_indices = []
        self.batch_indices.extend(
            [
                {"range": split, "record": k}
                for k, v in self.record_indices.items()
                for split in np.split(v, 300 / self.batch_size)
            ]
        )

    # def initialize(self):
    #     self.current_record_idx = -1
    #     self.set_next()

    # def set_next(self):
    #     self.current_record_idx += 1
    #     self.current_record = self.records[self.current_record_idx]
    #     self.current_position = 0
    # self.get_record()

    def split_data(self):
        n_records = len(self.records)
        n_eval = int(n_records * self.eval_ratio)
        n_train = n_records - n_eval
        self.shuffle_data()
        self.batch_data()
        train_data = STAGESSubset(self, np.arange(n_eval, n_records))
        eval_data = STAGESSubset(self, np.arange(0, n_eval), name="Eval")
        return train_data, eval_data

    # def get_record(self):
    #     # if not self.loaded_record:
    #     self.loaded_record = {
    #         "data": self.data[self.current_record]["data"],
    #         "target": self.data[self.current_record]["target"],
    #         "weights": self.data[self.current_record]["weights"],
    #         "record": self.current_record,
    #         "current_pos": 0,
    #     }
    # }
    # elif self.loaded_record['record'] != self.current_record:
    # self.
    # self.loaded_record = {k: v for k, v in zip(['data', 'target', 'weights'], load_h5_data(
    #     os.path.join(self.data_dir, self.current_record), seg_size=self.seg_size))}
    # self.loaded_record.update(
    #     {'record': self.current_record, 'current_pos': 0})
    # print('debug')

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
            current_record = self.batch_indices[idx]["record"]
            samples = self.batch_indices[idx]["range"]

            # from time import time

            # Grab data
            # start = time()
            with File(os.path.join(self.data_dir, current_record), "r") as f:
                # x = f['trainD'][samples].shape
                x = np.stack([f["trainD"][s] for s in samples]).astype("float32")
                t = np.stack([f["trainL"][s] for s in samples]).astype("float32").squeeze()
                w = np.stack([f["trainW"][s] for s in samples]).astype("float32")
            # print("Elapsed time: ", time() - start)

            # mask = [x for x in range(300) if x in samples]
            # start = time()
            # with File(os.path.join(self.data_dir, current_record), "r") as f:
            #     # x = f['trainD'][samples].shape
            #     x_ = f["trainD"][mask].astype("float32")
            #     t_ = f["trainL"][mask].astype("float32").squeeze()
            #     w_ = f["trainW"][mask].astype("float32")
            # print("Elapsed time: ", time() - start)

            # samples = sorted(samples)
            # start = time()
            # with File(os.path.join(self.data_dir, current_record), "r") as f:
            #     # x = f['trainD'][samples].shape
            #     x__ = f["trainD"][samples].astype("float32")
            #     t__ = f["trainL"][samples].astype("float32").squeeze()
            #     w__ = f["trainW"][samples].astype("float32")
            # print("Elapsed time: ", time() - start)

            # current_record = self.current_record
            # current_pos = self.current_position
            # samples = self.record_indices[current_record][current_pos:current_pos+self.batch_size]
            # x = self.data[current_record]["data"][samples]
            # t = self.data[current_record]["target"][samples]
            # x = self.loaded_record['data'][samples]
            # w = self.data[current_record]["weights"][samples]
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
            print("Bug")

        if np.isnan(x).any():
            print("NaNs detected!")
            x[np.isnan(x)] = 2 ** -23
            x[np.isinf(x)] = 2 ** -23

        return x, t, w, current_record, samples
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


class STAGESSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, record_indices, name="Train"):
        self.dataset = dataset
        self.record_indices = record_indices
        self.records = [self.dataset.records[idx] for idx in self.record_indices]
        self.batch_indices = [
            idx for idx, v in enumerate(self.dataset.batch_indices) for r in self.records if v["record"] == r
        ]
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
Records: {self.records[:10]} ... {self.records[10:]}
======================================
"""

        return s


class STAGESDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=None,
        data_dir=None,
        eval_ratio=None,
        n_jobs=None,
        n_workers=None,
        n_records=None,
        seg_size=None,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = {"train": os.path.join(data_dir, "train"), "test": os.path.join(data_dir, "test")}
        self.eval_ratio = eval_ratio
        self.n_jobs = n_jobs
        self.n_workers = n_workers
        self.n_records = n_records
        self.seg_size = seg_size
        self.dataset_params = dict(
            batch_size=self.batch_size,
            eval_ratio=self.eval_ratio,
            n_jobs=self.n_jobs,
            n_records=self.n_records,
            seg_size=self.seg_size,
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            dataset = STAGESDataset(data_dir=self.data_dir["train"], **self.dataset_params)
            self.train_data, self.eval_data = dataset.split_data()
        elif stage == "test":
            raise NotImplementedError
        else:
            raise ValueError(f"Supplied stage {stage} is not recognized.")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data, batch_size=None, shuffle=False, num_workers=self.n_workers, pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_data, batch_size=None, shuffle=False, num_workers=self.n_workers, pin_memory=True
        )

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        from argparse import ArgumentParser

        # DATASET specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        dataset_group = parser.add_argument_group("dataset")
        dataset_group.add_argument("--batch_size", default=30, type=int)
        dataset_group.add_argument("--data_dir", default="data/ssc_wsc/cc/5min", type=str)
        dataset_group.add_argument("--eval_ratio", default=0.1, type=float)
        dataset_group.add_argument("--n_jobs", default=-1, type=int)
        dataset_group.add_argument("--n_records", default=None, type=int)

        # DATALOADER specific
        dataloader_group = parser.add_argument_group("dataloader")
        dataloader_group.add_argument("--n_workers", default=0, type=int)

        return parser


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(add_help=False)
    parser = STAGESDataModule.add_dataset_specific_args(parser)
    args = parser.parse_args()
    args.n_records = 10
    dm = STAGESDataModule(**vars(args))
    dm.setup()
    train_loader = dm.train_dataloader()

    X, y, w, r, n = next(iter(train_loader))
    print("")
    print(f"Current record: {r}")
    print(f"Current sequence: {n}")
    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    print("w.shape: ", w.shape)

    dm_params = dict(
        batch_size=30,
        data_dir="data/ssc_wsc/cc/5min",
        eval_ratio=0.1,
        n_jobs=-1,
        n_records=100,
        n_workers=10,
        seg_size=60,
    )
    dm = STAGESDataModule(**dm_params)
    print(dm)
    dm.setup()
    print(dm.train_data)
    print(dm.eval_data)
    train_dl = dm.train_dataloader()
    pbar = tqdm(train_dl)
    for batch in pbar:
        pass
        # print("Hej")
