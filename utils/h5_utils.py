import numpy as np
from h5py import File


def load_h5_data(filename, seg_size):

    with File(filename, "r") as h5:
        # print(h5.keys())
        dataT = h5["trainD"][:].astype("float32")
        targetT = h5["trainL"][:].astype("float32")
        weights = h5["trainW"][:].astype("float32")

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

    return (
        np.reshape(dataT, [seq_in_file, n_segs, seg_size, -1]),
        np.reshape(targetT, [seq_in_file, n_segs, seg_size, -1]),
        np.reshape(weights, [seq_in_file, n_segs, seg_size]),
        seq_in_file,
    )
