import argparse
import json
import logging
import os
import random
import re
import sys
from glob import glob

import numpy as np
import pandas as pd
import scipy.io as sio
import skimage

from scipy import signal
from scipy.fft import fft
from scipy.fft import fftshift
from scipy.fft import ifft
from tqdm import tqdm

from utils.h5_utils import save_h5
from utils import edf_read_fns
from utils import load_scored_data

os.chdir("/home/users/alexno/sleep-staging")

cc_sizes = [2, 2, 4, 4, 0.4]
cc_overlap = 0.25

try:
    df = pd.read_csv("overview_file_cohortsEM-ling1.csv")
except FileNotFoundError:
    df = pd.read_csv("data_master.csv")

noiseM = sio.loadmat("preprocessing/noiseM.mat", squeeze_me=True, mat_dtype=False)["noiseM"]
meanV = noiseM["meanV"].item()
covM = noiseM["covM"].item()

# logger = logging.getLogger(__package__)

# Filter specifications for resampling from MATLAB
with open("utils/filter_coefficients/filter_specs.json", "r") as json_file:
    filter_specs = json.load(json_file)

with open("utils/channel_dicts/channel_names.txt", "r") as txt_file:
    ch_names = [re.split(", ", line.rstrip()) for line in txt_file]


def get_quiet_channel(channels, fs, meanV, covM):
    noise = np.zeros(len(channels))
    for idx, ch in enumerate(channels):
        noise[idx] = channel_noise_level(ch, fs, meanV, covM)
    return np.argmin(noise)


def channel_noise_level(channel, fs, meanV, covM):
    if isinstance(channel, np.ndarray):
        hjorth = extract_hjorth(channel, fs)
        noise_vec = np.zeros(hjorth.shape[1])
        for k in range(len(noise_vec)):
            M = hjorth[:, k][:, np.newaxis]
            x = M - meanV[:, np.newaxis]
            sigma = np.linalg.inv(covM)
            noise_vec[k] = np.sqrt(np.dot(np.dot(np.transpose(x), sigma), x))
        return np.mean(noise_vec)
    else:
        return np.inf


def get_alternative_names(ch, labels):

    for d in ch_names[ch]:
        synonym = [d == l for l in labels]
        if sum(synonym) > 0:
            break

    return synonym


# Use 5 minute sliding window.
def extract_hjorth(x, fs, dim=5 * 60, slide=5 * 60):

    # Length of first dimension
    dim = dim * fs

    # Overlap of segments in samples
    slide = slide * fs

    # Creates 2D array of overlapping segments
    # D = skimage.util.view_as_windows(x, dim, dim).T
    D = rolling_window_nodelay(x, dim, dim)
    D = np.delete(D, -1, axis=-1)

    # Extract Hjorth params for each segment
    dD = np.diff(D, 1, axis=0)
    ddD = np.diff(dD, 1, axis=0)
    mD2 = np.mean(D ** 2, axis=0)
    mdD2 = np.mean(dD ** 2, axis=0)
    mddD2 = np.mean(ddD ** 2, axis=0)

    top = np.sqrt(np.divide(mddD2, mdD2 + np.finfo(float).eps))

    mobility = np.sqrt(np.divide(mdD2, mD2 + np.finfo(float).eps))
    activity = mD2
    complexity = np.divide(top, mobility + np.finfo(float).eps)

    hjorth = np.array([activity, complexity, mobility])
    hjorth = np.log(hjorth + np.finfo(float).eps)
    return hjorth


def rolling_window_nodelay(vec, window, step):
    def calculate_padding(vec, window, step):
        import math

        N = len(vec)
        B = math.ceil(N / step)
        L = (B - 1) * step + window
        return L - N

    from skimage.util import view_as_windows

    pad = calculate_padding(vec, window, step)
    A = view_as_windows(np.pad(vec, (0, pad)), window, step).T
    zero_cols = pad // step
    return np.delete(A, np.arange(A.shape[1] - zero_cols, A.shape[1]), axis=1)


# @jit(nopython=True, parallel=True)
# @jit(nopython=True)
def encode_data(x1, x2, dim, slide, fs):

    # Length of the first dimension and overlap of segments
    dim = int(fs * dim)
    slide = int(fs * slide)

    # Create 2D array of overlapping segments
    zero_vec = np.zeros(dim // 2).astype(np.float32)
    input2 = np.concatenate((zero_vec, x2, zero_vec))
    D1 = rolling_window_nodelay(x1, dim, slide)
    D2 = rolling_window_nodelay(input2, dim * 2, slide)
    zero_mat = np.zeros((dim // 2, D1.shape[1]))
    D1 = np.concatenate([zero_mat, D1, zero_mat])

    keep_dims = D1.shape[1]
    D2 = D2[:, :keep_dims]
    D1 = D1[:, :keep_dims]

    # Fast implementation of auto/cross-correlation
    C = fftshift(
        np.real(
            ifft(
                fft(D1, dim * 2 - 1, axis=0, workers=-1) * np.conj(fft(D2, dim * 2 - 1, axis=0, workers=-1)), axis=0, workers=-2,
            )
        ),
        axes=0,
    ).astype(dtype=np.float32)

    # Remove mirrored part
    C = C[dim // 2 - 1 : -dim // 2]

    # Scale data with log modulus
    scale = np.log(np.max(np.abs(C) + 1, axis=0) / dim)
    try:
        C = C[..., :] / (np.amax(np.abs(C), axis=0) / scale)
        C[np.isnan(C)] == 0
        C[np.isinf(C)] == 0
    except RuntimeWarning:
        print("Error in log modulus scaling")

    return C


def load_signals(edf_file, fs, cohort, encoding):

    logging.info("\tLoading EDF header")
    temp_data, cFs, channel_labels = edf_read_fns[cohort](edf_file, fs)

    if encoding == "cc":  # TODO: this is broken
        # Resampling data. Filter coefficients are pulled from MATLAB
        data = [[]] * temp_data.shape[0] if not isinstance(temp_data, list) else [[]] * len(temp_data)
        for idx, orig_fs in enumerate(cFs):
            if orig_fs != fs:
                data[idx] = signal.upfirdn(
                    filter_specs[str(fs)][str(orig_fs)]["numerator"],
                    temp_data[idx],
                    filter_specs[str(fs)][str(orig_fs)]["up"],
                    filter_specs[str(fs)][str(orig_fs)]["down"],
                )
                if fs == 100:
                    if (
                        orig_fs == 256 or orig_fs == 512
                    ):  # Matlab creates a filtercascade which requires the 128 Hz filter be applied afterwards
                        data[idx] = signal.upfirdn(
                            filter_specs[str(fs)]["128"]["numerator"],
                            data[idx],
                            filter_specs[str(fs)]["128"]["up"],
                            filter_specs[str(fs)]["128"]["down"],
                        )
            else:
                data[idx] = temp_data[idx]
    elif encoding == "raw":
        logging.info(f"\tResampling to {fs} Hz")
        # Resample data using polyphase filtering
        data = [[]] * temp_data.shape[0] if not isinstance(temp_data, list) else [[]] * len(temp_data)
        for idx, orig_fs in enumerate(cFs):
            if isinstance(temp_data[idx], np.ndarray):
                if orig_fs != fs:
                    data[idx] = signal.resample_poly(temp_data[idx], fs, orig_fs)
                else:
                    data[idx] = temp_data[idx]

    # Trim
    for idx in range(len(data)):
        # 30 represents the epoch length most often used in standard hypnogram scoring.
        if isinstance(data[idx], np.ndarray):
            rem = len(data[idx]) % (fs * 30)
            # Otherwise, if rem == 0, the following results in an empty array
            if rem > 0:
                logging.info(f"\tTrimming ch {idx} from {len(data[idx])} to {len(data[idx]) - rem}")
                data[idx] = data[idx][:-rem]

    # Select channels based on noise levels
    central_list = [0, 1]
    occipital_list = [2, 3]
    descr = ["left", "right"]
    # Select Central channel, index 0 in meanV and covM
    keep_idx_central = get_quiet_channel([data[j] for j in central_list], fs, meanV[0], covM[0])
    logging.info(f"\tKeeping {descr[keep_idx_central]} central EEG channel")
    # Select occipital channel, index 1 in meanV and covM
    if not all([isinstance(data[o], list) for o in occipital_list]):  # Some cohorts do not have occipital
        keep_idx_occipital = get_quiet_channel([data[j] for j in occipital_list], fs, meanV[1], covM[1])
        logging.info(f"\tKeeping {descr[keep_idx_central]} occipital EEG channel")
    else:
        keep_idx_occipital = None
    # Select only kept channels
    data = np.concatenate(
        [
            v
            for v in [
                data[central_list[keep_idx_central]][np.newaxis, :],
                data[occipital_list[keep_idx_occipital]][np.newaxis, :] if keep_idx_occipital is not None else None,
                data[-3:],
            ]
            if v is not None
        ]
    )

    return data


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    logging.info(f"Creating directory: {dir_path}")


def process_single_file(current_file, fs, seq_len, overlap, cohort, encoding="cc"):

    logging.info("\tLoading hypnogram")
    hyp = load_scored_data(current_file.split(".")[0], cohort=cohort)
    if hyp is None:
        logging.info("\tUnable to find hypnogram file!")
        return
    logging.info(f"\tHypnogram length: {len(hyp)}")
    logging.info(f"\tHypnogram classes: {np.unique(hyp)}")
    # raise MissingHypnogramError(os.path.basename(current_file))

    logging.info("\tLoading signals from PSG file")
    sig = load_signals(current_file, fs, cohort, encoding)
    logging.info(f"\tPSG data shape: {sig.shape}")
    hyp = hyp[: len(sig[0]) // (fs * 30), :]

    if encoding == "cc":

        label = np.repeat(hyp, 120, axis=0)

        # Filter signals
        wH = 0.2 / (fs / 2)
        wL = 49 / (fs / 2)
        bH, aH = signal.butter(5, wH, btype="high", output="ba")
        bL, aL = signal.butter(5, wL, btype="low", output="ba")
        sig = signal.filtfilt(bH, aH, sig, padlen=3 * (max(len(bH), len(aH)) - 1))
        sig = signal.filtfilt(bL, aL, sig, padlen=3 * (max(len(bL), len(aL)) - 1)).astype(
            np.float32
        )  # These should match MATLAB output

        # Do encoding
        encodings = []
        for ch, cc_size in zip(sig, cc_sizes):
            encodings.append(encode_data(ch, ch, cc_size, cc_overlap, fs))
        encodings.append(encode_data(sig[2], sig[3], cc_sizes[2], cc_overlap, fs))

        # Trim excess
        C = np.concatenate([enc[:, : np.size(encodings[-1], 1)] for enc in encodings])
        label = label[: C.shape[1]]
        if label.shape[1] > 1:
            delete_idx = []
        else:
            delete_idx = np.where(label == 7)[0]
        C = np.delete(C, delete_idx, axis=1)
        label = np.delete(label, delete_idx, axis=0)
        if hyp.shape[1] > 1:
            delete_idx = []
        else:
            delete_idx = np.where(hyp == 7)[0]
        hyp = np.delete(hyp, delete_idx, axis=0)

        # Design labels
        labels = np.zeros((5,) + label.shape).astype(np.uint8)
        for j in range(5):
            labels[j, label == j + 1] = 1
        # index = rolling_window_nodelay(np.arange(C.shape[1]), seq_len, seq_len - overlap).T
        index = skimage.util.view_as_windows(np.arange(C.shape[1]), seq_len, seq_len - overlap)
        M = np.stack([C[:, j] for j in index], axis=0)
        L = np.stack([labels[:, j] for j in index], axis=0)

        # Adjust weights depending on stage
        if hyp.shape[1] == 1:
            hyp = np.repeat(hyp, 2)
            mask = np.zeros(hyp.shape)
            dhyp = np.abs(np.sign(np.diff(hyp)))
            mask[: len(dhyp)] = mask[: len(dhyp)] + dhyp
            mask[1 : len(dhyp) + 1] = mask[1 : len(dhyp) + 1] + dhyp
            mask = (1 - mask).astype(np.float32)
            weight = np.zeros(hyp.shape)
            weight[hyp == 1] = 1.5
            weight[hyp == 2] = 2
            weight[hyp == 3] = 1
            weight[hyp == 4] = 2.5
            weight[hyp == 5] = 2
            weight *= mask
            weight = np.repeat(weight, 60)

            W = np.stack([weight[j] for j in index], axis=0)
            Z = None
        else:
            W = None
            Z = None

    elif encoding == "raw":

        # Filter signals
        logging.info(f"\tFilter signal data")
        eegFilter = signal.butter(2, [0.3, 35], btype="bandpass", output="sos", fs=fs)
        emgFilter = signal.butter(4, 10, btype="highpass", output="sos", fs=fs)
        sig[:-1] = signal.sosfiltfilt(eegFilter, sig[:-1])
        sig[-1] = signal.sosfiltfilt(emgFilter, sig[-1])
        sig = sig.astype(np.float32)
        # fmt: off

        # Trim excess
        C = np.stack([rolling_window_nodelay(s, 30 * fs, 30 * fs) for s in sig])

        # Design labels
        logging.info(f'\tCreating one-hot encoding hypnogram')
        labels = np.zeros((5, hyp.shape[0], hyp.shape[1])).astype(np.uint32)
        if cohort in ['cfs', 'chat', 'mesa', 'mros', 'shhs']:
            for j in range(5):
                labels[j, hyp == j] = 1
        else:
            for j in range(5):
                labels[j, hyp == j + 1] = 1

        # Mask vector to account for unknown stages (7), or anything else that should be excluded in the loss calculations
        Z = np.full(hyp.shape, False, dtype=bool)
        Z[np.where(hyp == 7)] = True  # Unknown stage 7

        # Create a rolling window index vector (for overlapping windows)
        index = rolling_window_nodelay(np.arange(C.shape[-1]), seq_len, seq_len - overlap).T

        # Create stacked arrays
        logging.info('\tCreating stacked arrays')
        M = np.stack([C[:, :, j] for j in index], axis=0)
        M = np.swapaxes(M, 2, 3).reshape((M.shape[0], M.shape[1], -1))
        L = np.stack([labels[:, j] for j in index], axis=0).repeat(30, axis=2)  # (Number of sequences, number of classes, hypnogram value for each second in sequence)
        Z = np.stack([Z[j] for j in index], axis=0)

        W = None

        # fmt: on
    return M, L, W, Z, None, None


def get_all_filenames(data_dirs):
    listF = []
    for directory in data_dirs:
        logging.info(f"Looking for PSG files in {directory}...")
        listT = sorted(glob(os.path.join(directory, "*.[EeRr][DdEe][FfCc]"))) or sorted(
            glob(os.path.join(directory, "**/*.[EeRr][DdEe][FfCc]"))
        )
        logging.info(f"Found {len(listT)} files.")
        listF += listT
    return listF


def assign_files(listF, cohort, test=None, list_slice=None):
    not_listed = []
    listed_as_train = []
    listed_as_test = []
    logging.info("Assigning files (optionally based on data master)")
    for current_file in tqdm(listF):
        if cohort == "ihc":
            current_fid = os.path.basename(current_file).split(".")[0][:5]
            id_col = "ID_Ling"
        else:
            current_fid = os.path.basename(current_file).split(".")[0]
            id_col = "FileID"
        # Some of the KHC ID's have a prepended 0, this removes it
        if (cohort == "khc") and (df[df[id_col] == current_fid].empty and not df[df[id_col] == current_fid.lstrip("0")].empty):
            current_fid = current_fid.lstrip("0")
        # The subject is not in the overview file
        if df[df[id_col] == current_fid].empty:
            not_listed.append(current_file)
        elif (df[df[id_col] == current_fid]["Sleep scoring training data"] == 1).bool():
            # elif (not (df.query(f'ID == "{current_fid}"')["Sleep scoring training data"] == 1).bool()) or (df.query(f'ID_Ling == "{current_fid}"')["Sleep scoring training data"] == 1).bool():
            listed_as_train.append(current_file)
        # elif (df.query(f'ID == "{current_fid}"')["Sleep scoring test data"] == 1).bool() or (df.query()).bool():
        #     listed_as_test.append(current_file)
        else:
            listed_as_test.append(current_file)
            # logging.info(f"Hmm... Something is wrong with {current_file}!")
            # something_wrong.append(current_file)
    original_list = listF
    if test:
        if cohort in ["ihc", "dhc", "ahc"]:  # IHC data should only be placed in test
            listF = listed_as_test + not_listed
        else:
            if listed_as_test:
                listF = listed_as_test
            else:
                listF = not_listed
    else:
        listF = not_listed + listed_as_train

    return listF[list_slice] if list_slice else listF


def process_data(args):

    # Seed everything
    random.seed(42)
    np.random.seed(42)

    # Parse args
    fs = args.fs
    cohort = args.cohort
    out_dir = args.out_dir
    seq_len = args.seq_len
    overlap = args.overlap
    encoding = args.encoding

    # If multiple data folders are given (comma-delimiter)
    data_dirs = args.data_dir.split(",")

    # Create folders
    ensure_dir(args.log_dir)
    logging.info(f"Placing preprocessing logs in {args.log_dir}")
    logging.info(f"Saving H5 files to {out_dir}")
    ensure_dir(out_dir)

    # Make a list of all the files by looping over all available data-sources
    listF = get_all_filenames(data_dirs)

    # Get a curated list of files to process
    listF = assign_files(listF, cohort, args.test, args.slice)

    # Iterate over all files in list
    logging.info(f"Processing and saving {len(listF)} files")
    for i, filename in enumerate(tqdm(listF)):
        logging.info(f"Current file: {filename}")
        M, L, W, Z, _, _ = process_single_file(filename, fs, seq_len, overlap, cohort, encoding=encoding)

        if cohort in ["dcsm"]:
            save_name = os.path.join(out_dir, os.path.dirname(filename).split(os.path.sep)[-1] + ".h5")
        else:
            save_name = os.path.join(out_dir, os.path.basename(filename).split(".")[0] + ".h5")
        logging.info(f"\tSaving file to {save_name}")
        save_h5(save_name, M, L, W, Z)

    logging.info("Finished preprocessing all files!")
    return 0


if __name__ == "__main__":

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data_dir", type=str, required=True, help='Path to EDF data.')
    parser.add_argument('-o', "--out_dir", type=str, required=True, help='Where to store H5 files.')
    parser.add_argument('-c', '--cohort', type=str, required=True, help='Name of cohort.\nUse this to control logic regarding hypnogram and EDF loading routines.')
    parser.add_argument("--fs", type=int, default=100, help='Desired output sampling frequency.')
    parser.add_argument("--seq_len", type=int, default=1200, help='Length of sequences to store on disk.\nFor no encoding (raw), passing `--seq_len 10` will store sequences of 10 consequtive 30 s epochs on disk.')
    parser.add_argument("--overlap", type=int, default=400, help='Amount of overlap between sequences.\nPassing `--overlap 5` for no encoding (raw) will store sequences overlapping with 5 consequetive 30 s epochs.')
    parser.add_argument("--encoding", type=str, choices=["cc", "raw"], default="cc", help='Type of encoding.\n`raw` means no encoding. `cc` means cross-correlation encoding.')
    parser.add_argument('--mix', action="store_true", help='If passed, will create new H5 files containing mixes of already processed files.')
    parser.add_argument('--test', action="store_true", default=False, help='Flag to signal test data.\nThis will ignore passed overlap argument and set overlap to 0.')
    parser.add_argument('--slice', type=lambda s: slice(*[int(e) if e.strip() else None for e in s.split(":")]), help='Run over a selection of subjects only.\nEg. passing `--slice 10:20` will process subjects 10 through 19.')
    args = parser.parse_args()
    # fmt: on

    # Create logger
    args.log_dir = os.path.join("logs", "preprocessing", args.cohort, args.encoding)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=logging.INFO,
        datefmt="%I:%M:%S",
        handlers=[logging.FileHandler(os.path.join(args.log_dir, "preprocessing.log"), "w"), logging.StreamHandler()],
    )

    if args.test:
        args.subset = "test"
        args.overlap = 0
    else:
        args.subset = "train"

    logging.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logging.info("Settings:")
    logging.info("---------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == (len(vars(args)) - 1):
            logging.info(f"{k:>15}\t{v}\n")
        else:
            logging.info(f"{k:>15}\t{v}")

    if args.mix:
        mix_encodings(args)
    else:
        process_data(args)
