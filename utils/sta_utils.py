import os

import numpy as np


def load_hypnogram_ihc(sta_file):

    if not os.path.exists(sta_file):
        # THIS IS A TEMPORARY HACK
        head, tail = os.path.split(sta_file)
        tail = "".join(s.upper() if i in set([5, 6]) else s for i, s in enumerate(tail))

        sta_file = os.path.join(head, tail)
    if not os.path.exists(sta_file):
        return None
    with open(sta_file, "r") as fp:
        hyp = np.loadtxt(fp).astype(np.uint32)[:, np.newaxis]

    hypnogram = np.zeros(hyp.shape, dtype=np.uint32)
    hypnogram[hyp == 0] = 1
    hypnogram[hyp == 1] = 2
    hypnogram[hyp == 2] = 3
    hypnogram[hyp == 3] = 4
    hypnogram[hyp == 4] = 4
    hypnogram[hyp == 5] = 5
    hypnogram[hypnogram == 0] = 7

    return hypnogram


def load_hypnogram_jcts(sta_file):

    if not os.path.exists(sta_file):
        return None
    with open(sta_file, "r") as fp:
        hyp = np.loadtxt(fp).astype(np.uint32)[:, 1, np.newaxis]

    hypnogram = np.zeros(hyp.shape, dtype=np.uint32)
    hypnogram[hyp == 0] = 1
    hypnogram[hyp == 1] = 2
    hypnogram[hyp == 2] = 3
    hypnogram[hyp == 3] = 4
    hypnogram[hyp == 4] = 4
    hypnogram[hyp == 5] = 5
    hypnogram[hypnogram == 0] = 7

    return hypnogram


def load_hypnogram_dhc(sta_file):

    if not os.path.exists(sta_file):
        return None
    with open(sta_file, "r") as fp:
        hyp = np.loadtxt(fp).astype(np.uint32)[:, 1, np.newaxis]

    hypnogram = np.zeros(hyp.shape, dtype=np.uint32)
    hypnogram[hyp == 0] = 1
    hypnogram[hyp == 1] = 2
    hypnogram[hyp == 2] = 3
    hypnogram[hyp == 3] = 4
    hypnogram[hyp == 4] = 4
    hypnogram[hyp == 5] = 5
    hypnogram[hypnogram == 0] = 7

    return hypnogram


def load_hypnogram_khc(sta_file):

    try:
        with open(sta_file, "r") as fp:
            hyp = np.loadtxt(fp).astype(np.uint32)[:, 1, np.newaxis]
    except:
        _, tail = os.path.split(sta_file)
        sta_file = os.path.join("data", "khc", "hypnogram", tail)

    if not os.path.exists(sta_file):
        return None
    with open(sta_file, "r") as fp:
        hyp = np.loadtxt(fp).astype(np.uint32)[:, 1, np.newaxis]

    hypnogram = np.zeros(hyp.shape, dtype=np.uint32)
    hypnogram[hyp == 0] = 1
    hypnogram[hyp == 1] = 2
    hypnogram[hyp == 2] = 3
    hypnogram[hyp == 3] = 4
    hypnogram[hyp == 4] = 4
    hypnogram[hyp == 5] = 5
    hypnogram[hypnogram == 0] = 7

    return hypnogram


def load_hypnogram_default(sta_file):

    try:
        with open(sta_file, "r") as fp:
            _hyp = np.loadtxt(fp)[:, 1].astype(np.uint32)[:, np.newaxis]
    except ValueError:
        with open(sta_file, "r") as fp:
            _hyp = np.loadtxt(fp, delimiter=",")
            if _hyp.shape[1] == 6:  # This is for the multi-scorer case in ISRC
                _hyp = _hyp.astype(np.uint32)
            else:
                _hyp = _hyp[:, 1].astype(np.uint32)[:, np.newaxis]
    except FileNotFoundError:
        return None
    hyp = np.zeros(_hyp.shape, dtype=np.uint32)
    hyp[_hyp == 0] = 1
    hyp[_hyp == 1] = 2
    hyp[_hyp == 2] = 3
    hyp[_hyp == 3] = 4
    hyp[_hyp == 4] = 4
    hyp[_hyp == 5] = 5
    hyp[hyp == 0] = 7

    return hyp


hypnogram_read_fns = {
    "dhc": load_hypnogram_dhc,
    "jcts": load_hypnogram_jcts,
    "ihc": load_hypnogram_ihc,
    "wsc": load_hypnogram_default,
    "ssc": load_hypnogram_default,
    "khc": load_hypnogram_khc,
    "ahc": None,
    "stages-stnf": None,
    "stages-bogn": None,
    "stages-gs": None,
    "stages-gsdv": None,
    "stages-mayo": None,
    "stages-msmi": None,
    "stages-msnf": None,
    "stages-msqw": None,
    "stages-msth": None,
    "stages-mstr": None,
    "stages-stlk": None,
}


def load_scored_data(fileid, cohort=None):

    if hypnogram_read_fns[cohort] is None:
        return np.ones((3000, 1), dtype=np.uint32) * 7

    sta_file = fileid + ".STA"
    if not os.path.exists(sta_file):
        sta_file = os.path.join("data", cohort, "hypnogram", os.path.split(fileid)[1].split(".")[0] + ".STA")

    hyp = hypnogram_read_fns[cohort](sta_file)
    hyp = np.concatenate([hyp, np.ones((2000, hyp.shape[1]), dtype=np.uint32) * 7], axis=0)

    return hyp
