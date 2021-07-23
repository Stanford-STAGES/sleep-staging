import argparse
import json
import multiprocessing
import os
import random
import re
from glob import glob

import mne
import numpy as np
import pandas as pd
import scipy.io as sio
import skimage

from h5py import File
from scipy import signal
from scipy.fft import fft
from scipy.fft import fftshift
from scipy.fft import ifft
from tqdm import tqdm

from utils import edf_read_fns
from utils import load_scored_data
from utils.errors import MissingHypnogramError
from utils.errors import MissingSignalsError
from utils.errors import ReferencingError

os.chdir("/home/users/alexno/sleep-staging")
nthread = multiprocessing.cpu_count()

cc_sizes = [2, 2, 4, 4, 0.4]
cc_overlap = 0.25

try:
    df = pd.read_csv("overview_file_cohortsEM-ling1.csv")
except:
    df = pd.read_csv("data_master.csv")

noiseM = sio.loadmat("preprocessing/noiseM.mat", squeeze_me=True, mat_dtype=False)["noiseM"]
meanV = noiseM["meanV"].item()
covM = noiseM["covM"].item()

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
                fft(D1, dim * 2 - 1, axis=0, workers=-1) * np.conj(fft(D2, dim * 2 - 1, axis=0, workers=-1)),
                axis=0,
                workers=-2,
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

    header = mne.io.read_raw_edf(edf_file)
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
                data[idx] = data[idx][:-rem]

    # Select channels based on noise levels
    central_list = [0, 1]
    occipital_list = [2, 3]
    # Select Central channel, index 0 in meanV and covM
    keep_idx_central = get_quiet_channel([data[j] for j in central_list], fs, meanV[0], covM[0])
    # Select occipital channel, index 1 in meanV and covM
    if not all([isinstance(data[o], list) for o in occipital_list]):  # Some cohorts do not have occipital
        keep_idx_occipital = get_quiet_channel([data[j] for j in occipital_list], fs, meanV[1], covM[1])
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


def process_single_file(current_file, fs, seq_len, overlap, cohort, encoding="cc"):
    missing_hyp = []
    missing_sigs = []

    hyp = load_scored_data(current_file.split(".")[0], cohort=cohort)
    if hyp is None:
        raise MissingHypnogramError(os.path.basename(current_file))

    sig = load_signals(current_file, fs, cohort, encoding)
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
        eegFilter = signal.butter(2, [0.3, 35], btype="bandpass", output="sos", fs=fs)
        emgFilter = signal.butter(4, 10, btype="highpass", output="sos", fs=fs)
        sig[:-1] = signal.sosfiltfilt(eegFilter, sig[:-1])
        sig[-1] = signal.sosfiltfilt(emgFilter, sig[-1])
        sig = sig.astype(np.float32)
        # fmt: off

        # Trim excess
        C = np.stack([rolling_window_nodelay(s, 30 * fs, 30 * fs) for s in sig])

        # Design labels
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
        M = np.stack([C[:, :, j] for j in index], axis=0)
        M = np.swapaxes(M, 2, 3).reshape((M.shape[0], M.shape[1], -1))
        L = np.stack([labels[:, j] for j in index], axis=0).repeat(30, axis=2)  # (Number of sequences, number of classes, hypnogram value for each second in sequence)
        Z = np.stack([Z[j] for j in index], axis=0)

        W = None

        # fmt: on
    return M, L, W, Z, None, None


def process_data(args):

    random.seed(42)
    np.random.seed(42)

    data_stack = None

    # If multiple data folders are given (comma-delimiter)
    data_dirs = args.data_dir.split(",")

    fs = args.fs
    save_dir = args.save_dir
    seq_len = args.seq_len
    overlap = args.overlap
    encoding = args.encoding

    # Make a list of all the files by looping over all available data-sources
    listF = []
    for directory in data_dirs:
        # random.seed(12345)
        listT = sorted(glob(os.path.join(directory, "*.[EeRr][DdEe][FfCc]")))
        listF += listT

    not_listed = []
    listed_as_train = []
    listed_as_test = []
    something_wrong = []
    missing_hyp = []
    missing_sigs = []
    for current_file in listF:
        current_fid = os.path.basename(current_file).split(".")[0]
        if df.query(
            f'ID == "{current_fid}"'
        ).empty:  # The subject is not in the overview file and is automatically added to the train files
            not_listed.append(current_file)
        elif (df.query(f'ID == "{current_fid}"')["Sleep scoring training data"] == 1).bool():
            listed_as_train.append(current_file)
        elif (df.query(f'ID == "{current_fid}"')["Sleep scoring test data"] == 1).bool():
            listed_as_test.append(current_file)
            continue
        else:
            print(f"Hmm... Something is wrong with {current_file}!")
            something_wrong.append(current_file)
            continue
    list_files = listF
    listF = not_listed + listed_as_train
    # random.seed(12345)
    random.shuffle(listF)

    save_names = [os.path.join(save_dir, str(np.random.randint(10000000, 19999999)) + ".h5") for _ in range(1000)]

    # Run through all files and generate H5 files containing 300 5 min sequences
    i = -1
    pbar = tqdm(range(len(listF)))
    for i in pbar:
        # if i < 6:
        #     continue
        # while True:
        # i += 1
        if i < len(listF):
            current_file = listF[i]

            # if os.path.basename(current_file).split('.')[0] != 'A1039_3 164125':
            #     continue

            pbar.set_description(current_file)

            M, L, W, is_missing_hyp, is_missing_sigs = process_single_file(current_file, fs, seq_len, overlap, encoding)
            if is_missing_hyp:
                missing_hyp.append(current_file)
                continue
            elif is_missing_sigs:
                missing_sigs.append(current_file)
                continue

            if data_stack is None:
                data_stack = M
                label_stack = L
                weight_stack = W
            else:
                data_stack = np.concatenate([data_stack, M], axis=-1)
                label_stack = np.concatenate([label_stack, L], axis=-1)
                weight_stack = np.concatenate([weight_stack, W], axis=-1)

        if data_stack.shape[-1] > 900 | (i == len(listF) - 1 & data_stack.shape[-1] > 300):

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            # Shuffle the data
            ind = np.random.permutation(data_stack.shape[-1])
            data_stack = data_stack[:, :, ind]
            label_stack = label_stack[:, :, ind]
            weight_stack = weight_stack[:, ind]

            # Save to H5 file
            save_name = save_names.pop(0)
            with File(save_name, "w") as f:
                f.create_dataset("trainD", data=data_stack[:, :, :300])  # (data_stack.shape[0], seq_len, 300))
                f.create_dataset("trainL", data=label_stack[:, :, :300])  # (data_stack.shape[0], seq_len, 300))
                f.create_dataset("trainW", data=weight_stack[:, :300])  # (data_stack.shape[0], seq_len, 300))

            # Remove written data from the stack
            data_stack = np.delete(data_stack, range(300), axis=-1)
            label_stack = np.delete(label_stack, range(300), axis=-1)
            weight_stack = np.delete(weight_stack, range(300), axis=-1)

    if not os.path.exists("./txt"):
        os.mkdir("./txt")
    with open("txt/not_listed.txt", "w") as f:
        f.writelines(map(lambda x: x + "\n", not_listed))
    with open("txt/listed_as_train.txt", "w") as f:
        f.writelines(map(lambda x: x + "\n", listed_as_train))
    with open("txt/listed_as_test.txt", "w") as f:
        f.writelines(map(lambda x: x + "\n", listed_as_test))
    with open("txt/something_wrong.txt", "w") as f:
        f.writelines(map(lambda x: x + "\n", something_wrong))
    with open("txt/missing_hyp.txt", "w") as f:
        f.writelines(map(lambda x: x + "\n", missing_hyp))
    with open("txt/missing_sigs.txt", "w") as f:
        f.writelines(map(lambda x: x + "\n", missing_sigs))
    with open("txt/processed_files.txt", "w") as f:
        f.writelines(map(lambda x: x + "\n", listF))

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=400)
    parser.add_argument("--encoding", type=str, choices=["cc", "raw"], default="cc")
    args = parser.parse_args()

    if args.encoding == "raw":
        args.seq_len = 10
        args.overlap = 5

    process_data(args)
