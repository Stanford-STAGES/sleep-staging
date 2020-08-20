import argparse
import os
import random
import tempfile
from glob import glob

import numpy as np
import pandas as pd
from h5py import File
from tqdm import tqdm

from errors import MissingHypnogramError
from errors import MissingSignalsError
from errors import ReferencingError
from process_data import process_single_file


df = pd.read_csv("overview_file_cohortsEM-ling1.csv")


def batch_start_jobs(args):

    random.seed(42)
    np.random.seed(42)

    data_stack = None

    # If multiple data folders are given (comma-delimiter)
    data_dirs = args.data_dir.split(",")

    fs = args.fs
    encoding_type = args.encoding_type
    seq_len = args.seq_len
    overlap = args.overlap
    process_fn = batch_process_and_save
    # if args.encoding_type == 'cc':
    #     process_fn = batch_process_and_save_encoding
    # elif args.encoding_type == 'raw':
    #     process_fn = batch_process_and_save_raw
    if args.test:
        subset = "test"
        overlap = 0
    else:
        subset = "train"

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
        if df.query(f'ID == "{current_fid}"').empty and not df.query(f'ID == "{current_fid.lstrip("0")}"').empty:
            current_fid = current_fid.lstrip("0")
        # The subject is not in the overview file and is automatically added to the train files
        if df.query(f'ID == "{current_fid}"').empty:
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
    if args.test:
        if listed_as_test:
            listF = listed_as_test
        else:
            listF = not_listed
    else:
        listF = not_listed + listed_as_train

    if args.slice:
        listF = listF[args.slice]
    # random.seed(12345)
    # random.shuffle(listF)
    # n_per_batch = int(np.ceil(len(listF) / n_jobs))
    # split_list = [listF[i:i+n_per_batch] for i in range(0, len(listF), n_per_batch)]
    # test = ['A0005_4 175057', 'A0003_4 164611', 'A0013_6 182931', 'A0008_7 033111', 'A0009_4 171358', 'A0013_7 120409', 'A0008_4 171252', 'A0001_4 165907', 'A0001_5 180135']
    print(f"Submitting {len(listF)} jobs...")
    for current_file in listF:
        # print(current_file)
        # if not os.path.basename(current_file).split('.')[0] in test:
        #     continue
        content = f"""#!/bin/bash
#
#SBATCH --job-name="{os.path.basename(current_file)}"
#SBATCH -p mignot,owners,normal
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --output="/home/users/alexno/sleep-staging/batch/logs/{subset}/{encoding_type}/{os.path.basename(current_file)}.out"
#SBATCH --error="/home/users/alexno/sleep-staging/batch/logs/{subset}/{encoding_type}/{os.path.basename(current_file)}.err"
##################################################

source $PI_HOME/miniconda3/bin/activate
conda activate stages
cd $HOME/sleep-staging

python -c 'from batch_processing import {process_fn.__name__}; {process_fn.__name__}("{current_file}", {fs}, {seq_len}, {overlap}, "{subset}", "{encoding_type}")'
"""
        with tempfile.NamedTemporaryFile(delete=False) as j:
            j.write(content.encode())
        os.system("sbatch {}".format(j.name))

    print("All jobs have been submitted!")


def batch_process_and_save(current_file, fs, seq_len, overlap, subset, encoding_type="raw"):
    import traceback

    try:
        M, L, W, _, _ = process_single_file(current_file, fs, seq_len, overlap, encoding=encoding_type)
    except MissingHypnogramError as err:
        missing_path = f"./batch/{subset}/{encoding_type}/missing_hyp"
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    except MissingSignalsError as err:
        missing_path = f"./batch/{subset}/{encoding_type}/missing_sigs"
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    except ReferencingError as err:
        missing_path = f"./batch/{subset}/{encoding_type}/referencing"
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    except:
        missing_path = f"./batch/{subset}/{encoding_type}/errs"
        err = traceback.format_exc()
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1

    # Save to H5 file
    save_dir = f"./data/{subset}/{encoding_type}/individual_encodings"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = os.path.join(save_dir, os.path.basename(current_file).split(".")[0] + ".h5")
    chunks_M = (1,) + M.shape[1:]  # M.shape[1], M.shape[2])
    chunks_L = (1,) + L.shape[1:]  # (1, L.shape[1], L.shape[2])
    if W is not None:
        chunks_W = (1,) + W.shape[1:]
        # chunks_M = (M.shape[0], M.shape[1], 1)
        # chunks_L = (L.shape[0], L.shape[1], 1)
        # chunks_W = (W.shape[0], 1)
    with File(save_name, "w") as f:
        f.create_dataset("M", data=M, chunks=chunks_M)
        f.create_dataset("L", data=L, chunks=chunks_L)
        if W is not None:
            f.create_dataset("W", data=W, chunks=chunks_W)


def batch_process_and_save_raw(current_file, fs, seq_len, overlap, subset, encoding_type="raw"):
    import traceback

    try:
        M, L, W, is_missing_hyp, is_missing_sigs = process_single_file(current_file, fs, seq_len, overlap, encoding=encoding_type)
    except MissingHypnogramError as err:
        missing_path = f"./batch/{subset}/{encoding_type}/missing_hyp"
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    except MissingSignalsError as err:
        missing_path = f"./batch/{subset}/{encoding_type}/missing_sigs"
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    except ReferencingError as err:
        missing_path = f"./batch/{subset}/{encoding_type}/referencing"
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    except:
        missing_path = f"./batch/{subset}/{encoding_type}/errs"
        err = traceback.format_exc()
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    # if is_missing_hyp:
    #     if not os.path.exists('./batch/missing_hyp'):
    #         os.makedirs('./batch/missing_hyp')
    #     with open(f'./batch/missing_hyp/{os.path.basename(current_file)}.txt', 'w') as f:
    #         f.writelines(map(lambda x: x + '\n', []))
    #     return -1
    # elif is_missing_sigs:
    #     if not os.path.exists('./batch/missing_sigs'):
    #         os.makedirs('./batch/missing_sigs')
    #     with open(f'./batch/missing_sigs/{os.path.basename(current_file)}.txt', 'w') as f:
    #         f.writelines(map(lambda x: x + '\n', []))
    #     return -1

    # Save to H5 file
    save_dir = f"./data/{subset}/raw/individual_encodings"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = os.path.join(save_dir, os.path.basename(current_file).split(".")[0] + ".h5")
    with File(save_name, "w") as f:
        f.create_dataset("M", data=M, chunks=(1, M.shape[1], M.shape[2]))
        f.create_dataset("L", data=L, chunks=(1, L.shape[1], L.shape[2]))
        if W:
            f.create_dataset("W", data=W)


def batch_process_and_save_encoding(current_file, fs, seq_len, overlap, encoding_type="cc"):
    import traceback

    try:
        M, L, W, is_missing_hyp, is_missing_sigs = process_single_file(current_file, fs, seq_len, overlap)
    except MissingHypnogramError as err:
        missing_path = f"./batch/{encoding_type}/missing_hyp"
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    except MissingSignalsError as err:
        missing_path = f"./batch/{encoding_type}/missing_sigs"
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    except ReferencingError as err:
        missing_path = f"./batch/{encoding_type}/referencing"
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    except:
        err = traceback.format_exc()
        missing_path = f"./batch/{encoding_type}/errs"
        if not os.path.exists(missing_path):
            os.makedirs(missing_path)
        with open(os.path.join(missing_path, f"{os.path.basename(current_file)}.txt"), "w") as f:
            f.write(str(err))
        return -1
    # if is_missing_hyp:
    #     if not os.path.exists('./batch/missing_hyp'):
    #         os.makedirs('./batch/missing_hyp')
    #     with open(f'./batch/missing_hyp/{os.path.basename(current_file)}.txt', 'w') as f:
    #         f.writelines(map(lambda x: x + '\n', []))
    #     return -1
    # elif is_missing_sigs:
    #     if not os.path.exists('./batch/missing_sigs'):
    #         os.makedirs('./batch/missing_sigs')
    #     with open(f'./batch/missing_sigs/{os.path.basename(current_file)}.txt', 'w') as f:
    #         f.writelines(map(lambda x: x + '\n', []))
    #     return -1

    # Save to H5 file
    save_dir = "./data/individual_encodings"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_name = os.path.join(save_dir, os.path.basename(current_file).split(".")[0] + ".h5")
    with File(save_name, "w") as f:
        f.create_dataset("M", data=M)
        f.create_dataset("L", data=L)
        f.create_dataset("W", data=W)


def batch_mix_encodings(args):

    random.seed(42)
    np.random.seed(42)

    fs = args.fs
    encoding_dir = args.encoding_dir
    save_dir = args.save_dir
    seq_len = args.seq_len
    overlap = args.overlap
    data_stack = None

    # Make a list of all the files by looping over all available data-sources
    h5_files = sorted(glob(os.path.join(encoding_dir, "*.h5")))
    random.shuffle(h5_files)

    save_names = [os.path.join(save_dir, str(np.random.randint(10000000, 19999999)) + ".h5") for _ in range(10000)]
    orig_len = len(save_names)

    # Run through all files and generate H5 files containing 300 5 min sequences
    i = -1
    pbar = tqdm(range(len(h5_files)))
    for i in pbar:
        # if i < 6:
        #     continue
        # while True:
        # i += 1
        if i < len(h5_files):
            current_file = h5_files[i]

            # if os.path.basename(current_file).split('.')[0] != 'A1039_3 164125':
            #     continue

            pbar.set_description(current_file)

            def load_h5(filename):
                with File(filename, "r") as f:
                    M = f["M"][:].astype(np.float32)
                    L = f["L"][:].astype(np.float32)
                    W = f["W"][:].astype(np.float32)
                return M, L, W

            M, L, W = load_h5(current_file)
            # M, L, W, is_missing_hyp, is_missing_sigs = process_single_file(current_file, fs, seq_len, overlap)

            if data_stack is None:
                data_stack = M
                label_stack = L
                weight_stack = W
            else:
                data_stack = np.concatenate([data_stack, M], axis=-1)
                label_stack = np.concatenate([label_stack, L], axis=-1)
                weight_stack = np.concatenate([weight_stack, W], axis=-1)

        if data_stack.shape[-1] > 900 | (i == len(h5_files) - 1 & data_stack.shape[-1] > 300):

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            # Shuffle the data
            ind = np.random.permutation(data_stack.shape[-1])
            # while True:
            #     # from time import time
            #     start = time()
            #     a = data_stack[:, :, ind]
            #     print(time() - start)
            #     break
            # while True:
            #     # from time import time
            #     start = time()
            #     b = np.take(data_stack, ind, axis=-1)
            #     print(time() - start)
            #     break
            # data_stack = data_stack[:, :, ind]
            # label_stack = label_stack[:, :, ind]
            # weight_stack = weight_stack[:, ind]
            # data_stack = np.take(data_stack, ind, axis=-1)

            # Save to H5 file
            save_name = save_names.pop(0)
            pbar.write(f"Writing to file: {save_name}")
            with File(save_name, "w") as f:
                dset_data = f.create_dataset("trainD", (data_stack.shape[0], seq_len, 300), dtype="f4")
                dset_labels = f.create_dataset("trainL", (5, seq_len, 300), dtype="f4")
                dset_weights = f.create_dataset("trainW", (seq_len, 300), dtype="f4")
                dset_data[:] = np.take(data_stack, ind[:300], axis=-1)
                dset_labels[:] = np.take(label_stack, ind[:300], axis=-1)
                dset_weights[:] = np.take(weight_stack, ind[:300], axis=-1)
                # f.create_dataset('trainD', data=np.take(data_stack, ind[:300], axis=-1))# (data_stack.shape[0], seq_len, 300))
                # f.create_dataset('trainL', data=np.take(label_stack, ind[:300], axis=-1))#(data_stack.shape[0], seq_len, 300))
                # f.create_dataset('trainW', data=np.take(weight_stack, ind[:300], axis=-1))#(data_stack.shape[0], seq_len, 300))
                # f.create_dataset('trainD', data=data_stack[:, :, :300])# (data_stack.shape[0], seq_len, 300))
                # f.create_dataset('trainL', data=label_stack[:, :, :300])#(data_stack.shape[0], seq_len, 300))
                # f.create_dataset('trainW', data=weight_stack[:, :300])#(data_stack.shape[0], seq_len, 300))

            # Remove written data from the stack
            # data_stack = np.delete(data_stack, range(300), axis=-1)
            # label_stack = np.delete(label_stack, range(300), axis=-1)
            # weight_stack = np.delete(weight_stack, range(300), axis=-1)
            data_stack = np.delete(data_stack, ind[:300], axis=-1)
            label_stack = np.delete(label_stack, ind[:300], axis=-1)
            weight_stack = np.delete(weight_stack, ind[:300], axis=-1)

    # if not os.path.exists('./txt'):
    #     os.mkdir('./txt')
    # with open('txt/not_listed.txt', 'w') as f:
    #     f.writelines(map(lambda x: x + '\n', not_listed))
    # with open('txt/listed_as_train.txt', 'w') as f:
    #     f.writelines(map(lambda x: x + '\n', listed_as_train))
    # with open('txt/listed_as_test.txt', 'w') as f:
    #     f.writelines(map(lambda x: x + '\n', listed_as_test))
    # with open('txt/something_wrong.txt', 'w') as f:
    #     f.writelines(map(lambda x: x + '\n', something_wrong))
    # with open('txt/missing_hyp.txt', 'w') as f:
    #     f.writelines(map(lambda x: x + '\n', missing_hyp))
    # with open('txt/missing_sigs.txt', 'w') as f:
    #     f.writelines(map(lambda x: x + '\n', missing_sigs))
    # with open('txt/processed_files.txt', 'w') as f:
    #     f.writelines(map(lambda x: x + '\n', h5_files))

    print(f"Saved {orig_len - len(save_names)} files to disk.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str, default='/oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/WSC_EDF/')
    # parser.add_argument("--data_dir", type=str, default='/oak/stanford/groups/mignot/psg/SSC/APOE')
    # parser.add_argument("--data_dir", type=str, default='/oak/stanford/groups/mignot/psg/WSC_EDF/')
    parser.add_argument("--data_dir", type=str, default="/oak/stanford/groups/mignot/psg/Korea (KHC)/SOMNO")
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=400)
    parser.add_argument("--n_jobs", type=int, default=200)
    parser.add_argument("--encoding_type", type=str, default="cc", choices=["cc", "raw"])
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--slice", type=lambda s: slice(*[int(e) if e.strip() else None for e in s.split(":")]))
    args = parser.parse_args()

    if args.mix:
        args.encoding_dir = "./data/individual_encodings"
        args.save_dir = "./data/batch_encodings"
        batch_mix_encodings(args)
    else:
        batch_start_jobs(args)
        # batch_process_and_save("/oak/stanford/groups/mignot/psg/ISRC/AL_36_112108.edf", 100, 1200, 0, "test", "cc")
        # batch_process_and_save("/oak/stanford/groups/mignot/psg/SSC/APOE/SSC_1958_1.EDF", 128, 10, 0, "test", "raw")
    # batch_process_and_save_encoding('/oak/stanford/groups/mignot/psg/SSC/APOE/SSC_1558_1.EDF', 100, 1200, 400)
