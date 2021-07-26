import os

import numpy as np
import pandas as pd

from utils.parse_xml_nsrr import parse_hypnogram


STAGE_MAP = {"W": 1, "N1": 2, "N2": 3, "N3": 4, "REM": 5}


def load_hypnogram_ihc(hyp_file):

    if not os.path.exists(hyp_file):
        # THIS IS A TEMPORARY HACK
        head, tail = os.path.split(hyp_file)
        tail = "".join(s.upper() if i in set([5, 6]) else s for i, s in enumerate(tail))

        hyp_file = os.path.join(head, tail)
    if not os.path.exists(hyp_file):
        return None
    with open(hyp_file, "r") as fp:
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


def load_hypnogram_jcts(hyp_file):

    if not os.path.exists(hyp_file):
        return None
    with open(hyp_file, "r") as fp:
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


def load_hypnogram_dhc(hyp_file):

    if not os.path.exists(hyp_file):
        return None
    with open(hyp_file, "r") as fp:
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


def load_hypnogram_khc(hyp_file):

    try:
        with open(hyp_file, "r") as fp:
            hyp = np.loadtxt(fp).astype(np.uint32)[:, 1, np.newaxis]
    except:
        _, tail = os.path.split(hyp_file)
        hyp_file = os.path.join("data", "khc", "hypnogram", tail)

    if not os.path.exists(hyp_file):
        return None
    with open(hyp_file, "r") as fp:
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


def load_hypnogram_sta(fileid):

    sta_file = fileid + ".STA"
    # if not os.path.exists(sta_file):
    #     sta_file = os.path.join("data", cohort, "hypnogram", os.path.split(fileid)[1].split(".")[0] + ".STA")

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


def load_hypnogram_nsrr(hyp_file):

    xml_file = hyp_file.split(".")[0] + "-nsrr.xml"

    df_hypnogram = parse_hypnogram(xml_file, "xml")
    hypnogram = df_hypnogram["label"].values

    return np.asarray(hypnogram)[:, np.newaxis]


def load_hypnogram_ids(hyp_file):

    parts = hyp_file.split(".")
    if len(parts) > 1:
        parts = parts[0]
    dirname, basename = os.path.split(hyp_file)
    if basename == "hypnogram":
        hyp_file = os.path.join(dirname, basename + ".ids")
    else:
        hyp_file = os.path.join(dirname, "hypnogram.ids")

    df = pd.read_csv(hyp_file, header=None)
    dur = df[1].values // 30
    stages = df[2].values
    hypnogram = [STAGE_MAP[s] for (d, s) in zip(dur, stages) for _ in range(d)]

    return np.asarray(hypnogram)[:, np.newaxis]


hypnogram_read_fns = {
    "dhc": load_hypnogram_dhc,
    "jcts": load_hypnogram_jcts,
    "ihc": load_hypnogram_ihc,
    "wsc": load_hypnogram_sta,
    "ssc": load_hypnogram_sta,
    "khc": load_hypnogram_khc,
    "ahc": None,
    "cfs": load_hypnogram_nsrr,
    "chat": load_hypnogram_nsrr,
    "mesa": load_hypnogram_nsrr,
    "mros": load_hypnogram_nsrr,
    "shhs": load_hypnogram_nsrr,
    "dcsm": load_hypnogram_ids,
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

    if cohort not in set(hypnogram_read_fns.keys()):
        raise NotImplementedError

    if hypnogram_read_fns[cohort] is None:
        return np.ones((3000, 1), dtype=np.uint32) * 7

    hyp = hypnogram_read_fns[cohort](fileid)
    hyp = np.concatenate([hyp, np.ones((2000, hyp.shape[1]), dtype=np.uint32) * 7], axis=0)

    return hyp
