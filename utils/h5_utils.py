import os

import numpy as np
from h5py import File
from sklearn import preprocessing


SCALERS = {"robust": preprocessing.RobustScaler, "standard": preprocessing.StandardScaler}


def get_class_sequence_idx(hypnogram, selected_sequences):
    d = {
        "w": [idx for idx, hyp in enumerate(hypnogram) if (hyp == 0).any() and idx in selected_sequences],
        "n1": [idx for idx, hyp in enumerate(hypnogram) if (hyp == 1).any() and idx in selected_sequences],
        "n2": [idx for idx, hyp in enumerate(hypnogram) if (hyp == 2).any() and idx in selected_sequences],
        "n3": [idx for idx, hyp in enumerate(hypnogram) if (hyp == 3).any() and idx in selected_sequences],
        "r": [idx for idx, hyp in enumerate(hypnogram) if (hyp == 4).any() and idx in selected_sequences],
    }
    return d


def get_h5_info(filename):

    try:
        with File(filename, "r") as h5:
            dataT = h5["trainD"]
            seqs_in_file = dataT.shape[0]
    except OSError:
        seqs_in_file = None

    # print('Hej')

    return seqs_in_file, os.path.basename(filename)


def load_h5_data(filename, seg_size):

    with File(filename, "r") as h5:
        # print(h5.keys())
        dataT = h5["trainD"][:].astype("float32")
        targetT = h5["trainL"][:].astype("float32")
        weights = h5["trainW"][:].astype("float32")
        # dataT = h5["trainD"]
        # print("hej")

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
    # n_segs = dataT.shape[1] // seg_size
    n_segs = dataT.shape[-1] // seg_size

    # return (
    #     np.reshape(dataT, [seq_in_file, n_segs, seg_size, -1]),
    #     np.reshape(targetT, [seq_in_file, n_segs, seg_size, -1]),
    #     np.reshape(weights, [seq_in_file, n_segs, seg_size]),
    #     seq_in_file,
    # )
    return (
        np.reshape(dataT, [seq_in_file, -1, n_segs, seg_size]),
        np.reshape(targetT, [seq_in_file, -1, n_segs, seg_size]),
        np.reshape(weights, [seq_in_file, n_segs, seg_size]),
        seq_in_file,
    )


def load_psg_h5_data(filename, scaling=None):
    scaler = None

    if scaling:
        scaler = SCALERS[scaling]()

    with File(filename, "r") as h5:
        N, C, T = h5["M"].shape
        sequences_in_file = N

        if scaling:
            scaler.fit(h5["M"][:].transpose(1, 0, 2).reshape((C, N * T)).T)

    return sequences_in_file, scaler


def save_h5(save_name, M, L, W, Z):

    chunks_M = (1,) + M.shape[1:]  # M.shape[1], M.shape[2])
    chunks_L = (1,) + L.shape[1:]  # (1, L.shape[1], L.shape[2])
    if W is not None:
        chunks_W = (1,) + W.shape[1:]
    if Z is not None:
        chunks_Z = (1,) + Z.shape[1:]
    with File(save_name, "w") as f:
        f.create_dataset("M", data=M, chunks=chunks_M)
        f.create_dataset("L", data=L, chunks=chunks_L)
        if W is not None:
            f.create_dataset("W", data=W, chunks=chunks_W)
        if Z is not None:
            f.create_dataset("Z", data=Z, chunks=chunks_Z)


# def initialize_record(filename, scaling=None, overlap=True, adjustment=30):

#     if scaling in SCALERS.keys():
#         scaler = SCALERS[scaling]()
#     else:
#         scaler = None

#     with File(filename, "r") as h5:
#         M = h5["M"][:]
#         L = h5["L"][:]
#         N, C, T = M.shape
#         sequences_in_file = N

#         if scaler:
#             scaler.fit(M.transpose(1, 0, 2).reshape((C, N * T)).T)

#         # Remember that the output array from the H5 has 50 % overlap between segments.
#         # Use the following to split into even and odd
#         if overlap:
#             hyp_shape = L.shape
#             hyp_even = L[0::2]
#             hyp_odd = L[1::2]
#             stable_sleep = np.full([v for idx, v in enumerate(hyp_shape) if idx != 1], False)
#             stable_sleep[0::2] = get_stable_sleep_periods(hyp_even.argmax(axis=1), adjustment)[0]
#             stable_sleep[1::2] = get_stable_sleep_periods(hyp_odd.argmax(axis=1), adjustment)[0]
#         else:
#             if adjustment == 0:
#                 stable_sleep = np.full(L.argmax(axis=1).shape, True, dtype=np.bool)
#             else:
#                 stable_sleep = get_stable_sleep_periods(L.argmax(axis=1), adjustment)[0]

#         # Remove unknown stage
#         unknown_stage = get_unknown_stage(L)
#         stable_sleep[unknown_stage] = False

#     return sequences_in_file, scaler, stable_sleep


def initialize_record(filename, scaling=None, overlap=True, adjustment=30, sequence_length=5, balanced_sampling=False):

    if scaling in SCALERS.keys():
        scaler = SCALERS[scaling]()
    else:
        scaler = None

    with File(filename, "r") as h5:
        X = h5["M"][:]
        Y = h5["L"][:]
    N, C, T = X.shape
    hypnogram = Y[:, :, ::30]
    hyp_shape = hypnogram.shape
    sequences_in_file = N

    if scaler:
        scaler.fit(X.transpose(1, 0, 2).reshape((C, N * T)).T)

    # Remember that the output array from the H5 has 50 % overlap between segments.
    # Use the following to split into even and odd
    if overlap:
        hyp_even = hypnogram[0::2]
        hyp_odd = hypnogram[1::2]
        if adjustment > 0:
            stable_sleep = np.full([v for idx, v in enumerate(hyp_shape) if idx != 1], False)
            stable_sleep[0::2] = get_stable_sleep_periods(hyp_even.argmax(axis=1), adjustment)[0]
            stable_sleep[1::2] = get_stable_sleep_periods(hyp_odd.argmax(axis=1), adjustment)[0]
        else:
            stable_sleep = np.full([v for idx, v in enumerate(hyp_shape) if idx != 1], True)
    else:
        if adjustment > 0:
            stable_sleep = get_stable_sleep_periods(Y.argmax(axis=1), adjustment)[0]
        else:
            stable_sleep = np.full([v for idx, v in enumerate(hyp_shape) if idx != 1], True)

    # Remove unknown stage
    unknown_stage = get_unknown_stage(hypnogram)
    stable_sleep[unknown_stage] = False

    # Get bin counts
    if overlap:
        # hyp = h5["L"][::2].argmax(axis=1)[~get_unknown_stage(h5["L"][::2])][::30]
        hyp = hyp_even.argmax(axis=1)[~unknown_stage[::2] & stable_sleep[::2]]
    else:
        # hyp = h5["L"][:].argmax(axis=1)[~get_unknown_stage(h5["L"][:])][::30]
        hyp = hypnogram.argmax(axis=1)[~unknown_stage & stable_sleep]
    bin_counts = np.bincount(hyp, minlength=C)

    hypnogram = hypnogram.argmax(1)
    # select_sequences = np.where(stable_sleep.squeeze().any(axis=1))[0]
    # class_indices = get_class_sequence_idx(hypnogram, select_sequences)

    if isinstance(sequence_length, str) and sequence_length == "full":
        shape = hypnogram.shape
        hypnogram = (hypnogram.transpose(2, 0, 1).reshape(shape[2], -1).T)[np.newaxis]
        stable_sleep = (stable_sleep.transpose(2, 0, 1).reshape(shape[2], -1).T)[np.newaxis]
        sequences_in_file = hypnogram.shape[0]

    sorted_data = sort_record_data(os.path.basename(filename), hypnogram, scaler, stable_sleep, bin_counts, balanced_sampling)

    return sorted_data


def sort_record_data(record, hypnogram, scaler, stable_sleep, class_counts, balanced_sampling):
    select_sequences = np.where(stable_sleep.squeeze(-1).any(axis=1))[0]
    record_class_indices = get_class_sequence_idx(hypnogram, select_sequences)
    index_to_record = [{"record": record, "idx": x} for x in select_sequences]
    if balanced_sampling:
        index_to_record_class = {"w": [], "n1": [], "n2": [], "n3": [], "r": []}
        for c in index_to_record_class.keys():
            index_to_record_class[c] = [
                {
                    "idx": [idx for idx, i2r in enumerate(index_to_record) if i2r["idx"] == x and record == i2r["record"]][0],
                    "record": record,
                    "record_idx": x,
                }
                for x in record_class_indices[c]
            ]
    return dict(
        record_indices=(record, select_sequences),
        record_class_indices=(record, record_class_indices),
        index_to_record=index_to_record,
        index_to_record_class=index_to_record_class if balanced_sampling else None,
        scalers=(record, scaler),
        stable_sleep=(record, stable_sleep),
        cum_class_counts=(record, class_counts),
    )


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


def get_unknown_stage(onehot_hypnogram):
    return onehot_hypnogram.sum(axis=1) == 0
