import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xmltodict

from sleep_staging.utils.parse_xml_nsrr import parse_hypnogram

STAGE_MAP = {"W": 1, "N1": 2, "N2": 3, "N3": 4, "REM": 5}


def load_hypnogram_ssc(hyp_file):

    p = Path(hyp_file).parent
    hyp_id = Path(hyp_file).stem

    # HACk
    if "NARCO" in hyp_id:
        return load_hypnogram_sta(p.parent / "NARCO" / (hyp_id + ".STA"))

    try:
        hypnogram = load_hypnogram_sta(p / (hyp_id + ".STA"))
    except FileNotFoundError:
        file_id = list(p.glob(hyp_id + ".[Ee][Vv][Tt][Ss]*"))[0]
        hypnogram = load_hypnogram_evts(file_id)

    return hypnogram


# def load_hypnogram_cnc(hyp_file):

#     hyp_file_sta = [f for f in [f"{hyp_file}.STA", f"{hyp_file}.sta"] if os.path.isfile(f)]
#     hyp_file_xml = [
#         f
#         for f in [f"{hyp_file}.XML", f"{hyp_file}.xml", f"{hyp_file}.edf.xml", f"{hyp_file}.edf.XML",]
#         if os.path.isfile(f)
#     ]

#     if any(hyp_file_sta):
#         return load_hypnogram_sta(Path(hyp_file_sta[0]))
#     elif any(hyp_file_xml):
#         return load_hypnogram_cnc_xml(hyp_file_xml[0])
#     else:
#         return None


# def load_hypnogram_cnc_xml(xml_file):

#     with open(xml_file, "r", encoding="utf-8", errors="ignore") as f:
#         file = f.read()
#     annot = xmltodict.parse(file)
#     annot = annot["CMPStudyConfig"]["SleepStages"]
#     df = pd.DataFrame(annot).astype("int")
#     hyp = df["SleepStage"].values

#     hypnogram = np.zeros((hyp.shape[0], 1), dtype=np.uint32)
#     hypnogram[hyp == 0] = 1
#     hypnogram[hyp == 1] = 2
#     hypnogram[hyp == 2] = 3
#     hypnogram[hyp == 3] = 4
#     hypnogram[hyp == 4] = 4
#     hypnogram[hyp == 5] = 5
#     hypnogram[hypnogram == 0] = 7

#     return hypnogram


def load_hypnogram_cnc(hyp_file):

    hyp_file_txt = [
        f for f in ["{}-psg-txt.txt".format(hyp_file.split("-")[0]), "{}-PSG-txt.txt".format(hyp_file.split("-")[0])]
    ]
    hyp_file_sta = [f for f in ["{}.STA".format(hyp_file), "{}.sta".format(hyp_file)] if os.path.isfile(f)]
    hyp_file_xml = [
        f
        for f in [
            "{}.XML".format(hyp_file),
            "{}.xml".format(hyp_file),
            "{}.edf.xml".format(hyp_file),
            "{}.edf.XML".format(hyp_file),
        ]
        if os.path.isfile(f)
    ]

    if any(hyp_file_sta):
        return load_hypnogram_sta(Path(hyp_file_sta[0]))
    elif any(hyp_file_xml):
        return load_hypnogram_cnc_xml(hyp_file_xml[0])
    elif any(hyp_file_txt):
        return load_hypnogram_cnc_txt(hyp_file_txt[0])
    else:
        return None


def load_hypnogram_cnc_txt(txt_file):
    def isfloat(x):
        try:
            float(x)
            return True
        except:
            return False

    lookup = {"wake": ["ÐÑ¾õ"], "n1": ["1ÆÚ"], "n2": ["2ÆÚ"], "n3": ["3ÆÚ", "4ÆÚ"], "R": ["Rem"]}

    if not os.path.exists(txt_file):
        return None
    with open(txt_file, "rb") as fp:
        df = pd.read_csv(
            fp, encoding="unicode_escape", names=["idx", "stage", "start_time", "something"], delimiter="\t"
        )

    df = df[df["idx"].apply(lambda x: isfloat(x))]
    df["idx"] = df["idx"].apply(lambda x: int(x) - 1)

    hyp = np.zeros((df.shape[0], 1), dtype=np.uint32)
    hyp[df[df["stage"].isin(lookup["wake"])]["idx"]] = 1
    hyp[df[df["stage"].isin(lookup["n1"])]["idx"]] = 2
    hyp[df[df["stage"].isin(lookup["n2"])]["idx"]] = 3
    hyp[df[df["stage"].isin(lookup["n3"])]["idx"]] = 4
    hyp[df[df["stage"].isin(lookup["R"])]["idx"]] = 5
    hyp[hyp == 0] = 7

    hyp = np.repeat(hyp, 30, axis=0)

    return hyp


def load_hypnogram_cnc_xml(xml_file):

    with open(xml_file, "r", encoding="utf-8", errors="ignore") as f:
        file = f.read()
    annot = xmltodict.parse(file)
    annot = annot["CMPStudyConfig"]["SleepStages"]
    df = pd.DataFrame(annot).astype("int")
    hyp = df["SleepStage"].values

    hypnogram = np.zeros((hyp.shape[0], 1), dtype=np.uint32)
    hypnogram[hyp == 0] = 1
    hypnogram[hyp == 1] = 2
    hypnogram[hyp == 2] = 3
    hypnogram[hyp == 3] = 4
    hypnogram[hyp == 4] = 4
    hypnogram[hyp == 5] = 5
    hypnogram[hypnogram == 0] = 7

    hypnogram = np.repeat(hypnogram, 30, axis=0)

    return hypnogram


def load_hypnogram_ihc(hyp_file):

    try:
        directory, filename = os.path.split(hyp_file)  # only first 5 digits
        hyp_file = next(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if (f[-4:].lower() == ".sta") and (f[:5] == filename[:5])
        )
    except StopIteration:
        return None
    with open(hyp_file, "r") as fp:
        hyp = np.loadtxt(fp).astype(np.uint32)[:, np.newaxis]

    # if not os.path.exists(hyp_file):
    #     # THIS IS A TEMPORARY HACK
    #     head, tail = os.path.split(hyp_file)
    #     tail = "".join(s.upper() if i in set([5, 6]) else s for i, s in enumerate(tail))

    #     hyp_file = os.path.join(head, tail)
    # if not os.path.exists(hyp_file):
    #     return None
    # with open(hyp_file, "r") as fp:
    #     hyp = np.loadtxt(fp).astype(np.uint32)[:, np.newaxis]

    hypnogram = np.zeros(hyp.shape, dtype=np.uint32)
    hypnogram[hyp == 0] = 1
    hypnogram[hyp == 1] = 2
    hypnogram[hyp == 2] = 3
    hypnogram[hyp == 3] = 4
    hypnogram[hyp == 4] = 4
    hypnogram[hyp == 5] = 5
    hypnogram[hypnogram == 0] = 7

    hypnogram = np.repeat(hypnogram, 30, axis=0)

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


def load_hypnogram_dhc(hyp_file, light_epochs_filepath="/home/groups/mignot/sleep-staging/DHC_light_epochs.txt"):

    hyp_file = [f for f in [hyp_file, "{}.sta".format(hyp_file), "{}.STA".format(hyp_file)] if os.path.isfile(f)]

    if len(hyp_file) == 0:
        return None
    hyp = np.loadtxt(hyp_file[0]).astype(np.uint32)[:, 1, np.newaxis]
    directory, filename = os.path.split(hyp_file[0])  # only first 5 digits
    ID = filename.split(".")[0]

    df_epochs = pd.read_csv(light_epochs_filepath, header=0, delimiter=" ")

    # identify lights off and on.
    df_out = df_epochs[df_epochs["Name"].isin([ID])]
    assert len(df_out) == 1
    lights_off = int(df_out["Light_off_epoch"].values[0])
    lights_on = int(df_out["Light_on_epoch"].values[0])

    hypnogram = np.zeros(hyp.shape, dtype=np.uint32)
    hypnogram[hyp == 0] = 1
    hypnogram[hyp == 1] = 2
    hypnogram[hyp == 2] = 3
    hypnogram[hyp == 3] = 4
    hypnogram[hyp == 4] = 4
    hypnogram[hyp == 5] = 5
    hypnogram[:lights_off] = 7
    hypnogram[lights_on:] = 7
    hypnogram[hypnogram == 0] = 7

    hypnogram = np.repeat(hypnogram, 30, axis=0)

    return hypnogram


# def load_hypnogram_khc(hyp_file):

#     try:
#         with open(hyp_file, "r") as fp:
#             hyp = np.loadtxt(fp).astype(np.uint32)[:, 1, np.newaxis]
#     except:
#         _, tail = os.path.split(hyp_file)
#         hyp_file = os.path.join("data", "khc", "hypnogram", tail)

#     if not os.path.exists(hyp_file):
#         return None
#     with open(hyp_file, "r") as fp:
#         hyp = np.loadtxt(fp).astype(np.uint32)[:, 1, np.newaxis]


def load_hypnogram_khc(hyp_file, exts=["", ".sta"]):

    try:
        directory, filename = os.path.split(hyp_file)
        hyp_file = next(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower() in ["{}{}".format(filename.lower(), ext.lower()) for ext in exts]
        )
    except StopIteration:
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

    hypnogram = np.repeat(hypnogram, 30, axis=0)

    return hypnogram


def load_hypnogram_sta(fileid: Path):

    if not isinstance(fileid, Path):
        fileid = Path(fileid)
    # sta_file = fileid + ".STA"
    if ".sta" not in fileid.suffix.lower():
        sta_file = fileid.with_suffix(".STA")
    else:
        sta_file = fileid
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

    hyp = np.repeat(hyp, 30, axis=0)

    return hyp


def load_hypnogram_evts(hyp_file):

    # dictionary
    wake = ["wake", "w", '"wake"', 0.0, "0"]
    rem = ["r", "rem", '"rem"', 5.0, "5"]
    n1 = ["n1", "stage 1", '"stage 1"', 1.0, "1"]
    n2 = ["n2", "stage 2", '"stage 2"', 2.0, "2"]
    n3 = ["n3", "stage 3", '"stage 3"', "n4", "stage 4", '"stage 4"', 3.0, 4.0, "3", "4"]

    def timestr_2_timeidx(timestr):
        """assummed timestr format: HH:MM:SS:FFF or float"""

        if len(timestr.split(":")) > 1:
            timestr_ = timestr.split(":")
            return int(float(timestr_[0]) * 3600 + float(timestr_[1]) * 60 + float(timestr_[2]))
        else:
            return int(float(timestr))

    def estimate_sample_rate(x0, x1, y0, y1, possible_sample_rates=[256, 512]):

        if y1 < y0:
            y1 += 24 * 60 * 60
        sample_rate = round((x1 - x0) / (y1 - y0))
        assert sample_rate in possible_sample_rates
        return sample_rate

    if not os.path.exists(hyp_file):
        return None
    with open(hyp_file, "r") as f:
        try:
            df = pd.read_csv(
                f, delimiter=",", names=["Start Sample", "End Sample", "Start Time", "End Time", "Event", "File Name"]
            )
        except:
            return None

        # remove non-stage stuff
        df = df[df["Event"].str.lower().isin(wake + rem + n1 + n2 + n3)]

        # to be removed
        if len(df) == 0:
            return None

        # estimate sample rate:
        sample = df[["Start Sample", "Start Time"]].sample(2).sort_index()
        sample_rate = estimate_sample_rate(
            x0=int(sample["Start Sample"].iloc[0]),
            x1=int(sample["Start Sample"].iloc[1]),
            y0=timestr_2_timeidx(sample["Start Time"].iloc[0]),
            y1=timestr_2_timeidx(sample["Start Time"].iloc[1]),
        )

        df["time_idx"] = df.apply(lambda x: ((int(x["Start Sample"])) // (30 * sample_rate)), axis=1)

    hypnogram = np.zeros((df["time_idx"].max() + 1, 1), dtype=np.uint32)
    hypnogram[df[df["Event"].str.lower().isin(wake)]["time_idx"].values] = 1
    hypnogram[df[df["Event"].str.lower().isin(n1)]["time_idx"].values] = 2
    hypnogram[df[df["Event"].str.lower().isin(n2)]["time_idx"].values] = 3
    hypnogram[df[df["Event"].str.lower().isin(n3)]["time_idx"].values] = 4
    hypnogram[df[df["Event"].str.lower().isin(rem)]["time_idx"].values] = 5
    hypnogram[hypnogram == 0] = 7

    # print(hypnogram)

    hypnogram = np.repeat(hypnogram, 30, axis=0)

    return hypnogram


def load_hypnogram_nsrr(hyp_file):

    hyp_file = Path(hyp_file)
    file_id = hyp_file.stem
    xml_extension = "-nsrr.xml"
    xml_file = hyp_file.parent / (hyp_file.name + xml_extension)
    if not xml_file.exists():
        while not list(xml_file.parent.rglob(f"*{file_id + xml_extension}")):
            xml_file = xml_file.parent
        xml_file = list(xml_file.parent.rglob(f"*{file_id + xml_extension}"))[0]

    df_hypnogram = parse_hypnogram(xml_file, "xml")
    hypnogram = np.asarray(df_hypnogram["label"].values)[:, np.newaxis]

    return np.repeat(hypnogram, 30, axis=0)


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

    hypnogram = np.repeat(np.asarray(hypnogram)[:, np.newaxis], 30, axis=0)

    return hypnogram


def load_hypnogram_fhc(hyp_file, exts=["", "hypnoexp.txt"]):

    wake = ["v", "w"]
    rem = ["sp", "rem"]
    n1 = ["s1", "n1"]
    n2 = ["s2", "n2"]
    n3 = ["s3", "s4", "n3", "n4"]

    try:
        directory, filename = os.path.split(hyp_file)
        hyp_file = next(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower() in ["{}{}".format(filename.lower(), ext.lower()) for ext in exts]
        )
    except StopIteration:
        return None

    df = pd.read_csv(
        hyp_file, delimiter="\t", header=None, names=["time_since_start", "timestr", "hyp_label", "hyp_idx"]
    )
    if len(df) == 0:
        return None
    df["hyp_label"] = df["hyp_label"].apply(lambda x: x.split(" ")[0])
    df["epoch_idx"] = df.apply(lambda x: (x["time_since_start"] // 30) - 1, axis=1)

    hypnogram = np.zeros((df["epoch_idx"].max() + 1, 1), dtype=np.uint32)
    hypnogram[df[df["hyp_label"].str.lower().isin(wake)]["epoch_idx"].values] = 1
    hypnogram[df[df["hyp_label"].str.lower().isin(n1)]["epoch_idx"].values] = 2
    hypnogram[df[df["hyp_label"].str.lower().isin(n2)]["epoch_idx"].values] = 3
    hypnogram[df[df["hyp_label"].str.lower().isin(n3)]["epoch_idx"].values] = 4
    hypnogram[df[df["hyp_label"].str.lower().isin(rem)]["epoch_idx"].values] = 5
    hypnogram[hypnogram == 0] = 7

    hypnogram = np.repeat(hypnogram, 30, axis=0)

    return hypnogram

    # try:
    #     directory, filename = os.path.split(hyp_file)
    #     hyp_file = next(
    #         os.path.join(directory, f)
    #         for f in os.listdir(directory)
    #         if f.lower() in ["{}{}".format(filename.lower(), ext.lower()) for ext in exts]
    #     )

    # except StopIteration:
    #     return None
    # with open(hyp_file, "r") as f:
    #     df = pd.read_csv(f, delimiter="\t", header=None, names=["time_since_start", "timestr", "hyp_label", "hyp_idx"])
    #     if len(df) == 0:
    #         return None
    #     else:
    #         df["time_idx"] = df.apply(lambda x: (x["time_since_start"] // 30) - 1, axis=1)

    # hypnogram = np.zeros((df["time_idx"].max() + 1, 1), dtype=np.uint32)
    # hypnogram[df[df["hyp_idx"].isin([1])]["time_idx"].values] = 1
    # hypnogram[df[df["hyp_idx"].isin([3])]["time_idx"].values] = 2
    # hypnogram[df[df["hyp_idx"].isin([4])]["time_idx"].values] = 3
    # hypnogram[df[df["hyp_idx"].isin([5])]["time_idx"].values] = 4
    # hypnogram[df[df["hyp_idx"].isin([6])]["time_idx"].values] = 4
    # hypnogram[df[df["hyp_idx"].isin([2])]["time_idx"].values] = 5
    # hypnogram[df[df["hyp_idx"].isin([0])]["time_idx"].values] = 7
    # hypnogram[hypnogram == 0] = 7

    # return hypnogram

def load_hypnogram_csv(fileid):

    # assert(fileid[-4:].lower() == ".csv")

    if not isinstance(fileid, Path):
        fileid = Path(fileid)
    
    if not '.csv' in fileid.suffix.lower():
        fileid = fileid.with_suffix('.csv')
    
    if not fileid.exists():
        return None

    print(f'Loading {fileid}')

    clean_string = lambda x: re.sub(r'[^a-zA-Z0-9]', '', x).lower()
    time_format = "%H:%M:%S"
    wake = ['wake']
    rem = ['rem']
    n1 = ['stage1']
    n2 = ['stage2']
    n3 = ['stage3', 'stage4']
    label_2_idx = {'wake': 1, 'stage1': 2, 'stage2': 3, 'stage3': 4, 'stage4': 4, 'rem': 5}

    # read hypnograms
    # try:
    #     df = pd.read_csv(fileid, skiprows=1, sep=None, usecols=[0, 1, 2], names=['Start Time', 'Duration (seconds)', 'Event']).dropna(0)
    # except Exception:
    df = pd.read_csv(fileid, skiprows=1, usecols=[0, 1, 2], names=['Start Time', 'Duration (seconds)', 'Event']).dropna(0)
    # while True:
    #     df = pd.read_csv(fileid, names=['Start Time', 'Duration (seconds)', 'Event', 'comment'])
    #     if 
    
    # Remove any NaN containing rows
    # df.dropna(axis=0)

    if df.empty:
        print("Dataframe does not contain any sleep stages")
        return None
    
    if len(df) < 10:
        print(f'Subject {fileid.stem} has no hypnogram')
        return None
    
    # Create "Relative starting time" and "Duration" for events
    # -------------   
    df['Event'] = df['Event'].apply(lambda x: clean_string(x))
    
    # Convert timestamps to datetime objects
    timestamps_dt = [datetime.strptime(ts, time_format) for ts in df['Start Time']]

    # Account for change of day - we assume a chronoligical series
    for i in range(1, len(timestamps_dt)):
        if timestamps_dt[i] < timestamps_dt[i-1]:
            timestamps_dt[i] += timedelta(days=1)
    
    # set start time - we could setup this to be changed from outside
    start_time = timestamps_dt[0]

    # Calculate relative time
    df['Start Time relative'] = [(ts - start_time).total_seconds() for ts in timestamps_dt]
    try:
        assert(all(np.diff(df['Start Time relative'].values) >= 0)) # non-chronological order. Consider sorting...
    except AssertionError:
        print('Sorting df containing hypnogram...')
        df = df.sort_values('Start Time relative')
    
    # Select sleep stages only, and set to 30 second duration
    df_ = df[df['Event'].isin(wake + rem + n1 + n2 + n3)].copy()
    if df_.empty:
        print("Dataframe does not contain any sleep stages")
        return None
    if len(df_) < 10:
        print(f'Subject {fileid.stem} has hypnogram')
        return None
    df_['Event_idx'] = df_['Event'].map(label_2_idx).astype(int)
    # df_['Duration (seconds)'] = 30

    # Subject specific changes (this is a major HACK)
    if fileid.stem in ['MSQW00001']:
        assert df_.iloc[11]['Start Time relative'] == 7513
        df_ = df_.drop([df_.index[11]])
    if fileid.stem in ['STLK00096']:
        assert df_.iloc[64]['Start Time relative'] == 2261
        df_ = df_.drop([df_.index[64]])
    if fileid.stem in ['MSTH00018']:
        assert df_.iloc[140]['Start Time relative'] == 18159
        df_ = df_.drop([df_.index[140]])
    if 'MSQW' in fileid.stem:
        df_ = df_.drop(df_.loc[df_['Duration (seconds)'] == 0.0].index)
    if not df_.loc[df_['Duration (seconds)'] == 0.0].shape == df_.shape:
        df_ = df_.drop(df_.loc[df_['Duration (seconds)'] == 0.0].index)

    # Create hypnogram
    # -----------------
    recording_duration_sec = int(df_['Start Time relative'].iloc[-1] + df_['Duration (seconds)'].iloc[-1])
    if not df_['Duration (seconds)'].iloc[-1] > 0:
        recording_duration_sec += 30
    # try:
    #     assert(recording_duration_sec % 30 == 0)
    # except AssertionError:
    #     return None
    # hypnogram = 7 * np.ones((recording_duration_sec // 30, 1), dtype=np.uint32)
    hypnogram = 7 * np.ones((recording_duration_sec, 1), dtype=np.uint32)
    for df_idx in range(df_.shape[0]):
        # idx_start = int(event['Start Time relative'] // 30)
        # idx_stop = int(event['Duration (seconds)'] // 30) + idx_start
        event = df_.iloc[df_idx]
        idx_start = int(event['Start Time relative'])
        if df_idx < df_.shape[0] - 1:
            next_event = df_.iloc[df_idx + 1]
            idx_stop = int(next_event['Start Time relative'])
        else: 
            idx_stop = int(idx_start + event['Duration (seconds)']) if event['Duration (seconds)'] > 0 else int(idx_start + 30)
        # idx_stop = int(event['Duration (seconds)']) + idx_start
        assert((idx_stop - idx_start) % 30 == 0)
        stage = event['Event_idx']
        hypnogram[idx_start : idx_stop] = stage
    # [hypnogram.extend([s['Event_idx']] * int(s['Duration (seconds)'] / 30)) for _, s in df_.iterrows()]

    return hypnogram

    # return np.asarray(hypnogram)[:, np.newaxis]
    
    # recording_duration_sec = int(df_['Start Time relative'].iloc[-1] + df_['Duration (seconds)'].iloc[-1])
    # assert(recording_duration_sec % 30 == 0)

    # hypnogram = np.zeros((recording_duration_sec, 1), dtype=np.uint32)
    
    # # iterate over events
    # for n, event in df_.iterrows():
    #     hypnogram[int(event['Start Time relative']) : int(event['Start Time relative'] + event['Duration (seconds)'])] = event['Event_idx']
    
    # # set unassigned to 7
    # hypnogram[hypnogram == 0] = 7
    
    # # Reshape the array to create groups of 30 along the first axis --> median
    # hypnogram_resampled = np.median(hypnogram.reshape(-1, 30, hypnogram.shape[1]), axis=1).astype(int)

    # return hypnogram_resampled

hypnogram_read_fns = {
    "dhc": load_hypnogram_dhc,
    "jcts": load_hypnogram_jcts,
    "ihc": load_hypnogram_ihc,
    "wsc": load_hypnogram_sta,
    "ssc": load_hypnogram_ssc,
    "ssc_apoe": load_hypnogram_ssc,
    "ssc_narco": load_hypnogram_ssc,
    "khc": load_hypnogram_khc,
    "2ahc": None,
    "ahc": None,
    "fhc": load_hypnogram_fhc,
    "cnc": load_hypnogram_cnc,
    "cfs": load_hypnogram_nsrr,
    "chat": load_hypnogram_nsrr,
    "mesa": load_hypnogram_nsrr,
    "mros": load_hypnogram_nsrr,
    "shhs": load_hypnogram_nsrr,
    "dcsm": load_hypnogram_ids,
    "stages": load_hypnogram_csv,
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


def load_scored_data(fileid: str, cohort: Optional[str] = None):

    if cohort not in set(hypnogram_read_fns.keys()):
        return np.ones((100000, 1), dtype=np.uint32) * 7

    if hypnogram_read_fns[cohort] is None:
        return np.ones((100000, 1), dtype=np.uint32) * 7

    hyp = hypnogram_read_fns[cohort](fileid)
    if hyp is None:
        return np.ones((100000, 1), dtype=np.uint32) * 7
    hyp = np.concatenate([hyp, np.ones((60000, hyp.shape[1]), dtype=np.uint32) * 7], axis=0)

    return hyp
