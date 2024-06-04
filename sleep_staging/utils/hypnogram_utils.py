import numpy as np
from h5py import File

from sklearn.preprocessing import RobustScaler, StandardScaler

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
