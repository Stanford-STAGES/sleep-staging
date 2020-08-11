import random
import time

import numpy as np
from h5py import File
from joblib import Parallel, delayed
from tqdm import tqdm


def text_progressbar(seq, total=None):
    step = 1
    tick = time.time()
    while True:
        time_diff = time.time() - tick
        avg_speed = time_diff / step
        total_str = 'of %n' % total if total else ''
        print('step', step, '%.2f' % time_diff, 'avg: %.2f iter/sec' % avg_speed, total_str)
        step += 1
        yield next(seq)


all_bar_funcs = {'tqdm': lambda args: lambda x: tqdm(x, **args),
                 'txt': lambda args: lambda x: text_progressbar(x, **args),
                 'False': lambda args: iter,
                 'None': lambda args: iter}


def ParallelExecutor(use_bar='tqdm', **joblib_args):

    def aprun(bar=use_bar, **tq_args):

        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError('Value %s not supported as bar type' % bar)
            return Parallel(**joblib_args)(bar_func(op_iter))

        return tmp

    return aprun


def load_h5_data(filename, seg_size):

    with File(filename, 'r') as h5:
        # print(h5.keys())
        dataT = h5['trainD'][:].astype('float32')
        targetT = h5['trainL'][:].astype('float32')
        weights = h5['trainW'][:].astype('float32')

    # Hack to make sure axis order is preserved
    if dataT.shape[-1] == 300:
        dataT = np.swapaxes(dataT, 0, 2)
        targetT = np.swapaxes(targetT, 0, 2)
        weights = weights.T

    # print(dataT.shape)
    # print(targetT.shape)
    # print(weights.shape)
    # print(f'{filename} loaded - Training')

    seq_in_file = dataT.shape[0]
    n_segs = dataT.shape[1] // seg_size

    return (np.reshape(dataT, [seq_in_file, n_segs, seg_size, -1]),
            np.reshape(targetT, [seq_in_file, n_segs, seg_size, -1]),
            np.reshape(weights, [seq_in_file, n_segs, seg_size]), seq_in_file)
