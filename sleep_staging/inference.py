import argparse
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from rich.console import Console
from rich.logging import RichHandler
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from sleep_staging.preprocessing.process_data import process_single_file
from sleep_staging.utils.model_utils import get_model_from_ckpt

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(console=Console(width=255))])
logger = logging.getLogger("rich")


def check_datafile(data_file: Path):

    df = pd.read_csv(data_file, delimiter=";")
    df = df.loc[(df["Include"] == "Yes") & (df["Type"] == "PSG") & (df["Directory"].notnull())]

    file_exist = []
    file_noexist = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        directory = Path(row["Directory"])
        filename = row["OakFileName"]
        full_path = directory / filename
        data_files = list(
            set(directory.glob(f"{filename}.[eErR][dDeE][fFcC]"))
            | set(directory.glob(f"{filename.lower()}*.[eErR][dDeE][fFcC]"))
        )
        if len(data_files) == 1 and data_files[0].is_file():
            file_exist.append(full_path)
        else:
            file_noexist.append(full_path)

    logger.info(f"{len(file_exist)} matching files found")
    logger.info(f"{len(file_noexist)} files not found")


def get_datapaths(
    data_path: Optional[Path] = None,
    data_file: Optional[Path] = None,
    cohort: Optional[str] = None,
    pattern: Optional[str] = None,
) -> List[Path]:
    assert (data_path is None and data_file is not None) or (
        data_path is not None and data_file is None
    ), f"Please supply only one of arguments data_path and data_file, received data_path={data_path}, data_file={data_file}"

    if data_path is not None:
        if data_path.is_dir():
            # data_list = sorted(list([x for x in data_path.iterdir()]))
            data_list = sorted(list(data_path.rglob("*.[EeRr][DdEe][FfCc]")))
        else:
            data_list = []

        if pattern is not None:
            # data_list = [p for p in data_list if pattern in p.stem]
            data_list = [
                p for p in data_list if bool(re.search(pattern, p.stem) and p.suffix.lower() in [".edf", ".rec"])
            ]

        df = [None] * len(data_list)
    else:
        # Determine file extension
        assert data_file.suffix == ".csv", f"Please supply a data file in .csv format, received data_file={data_file}"

        df = pd.read_csv(data_file, delimiter=";").reset_index(drop=True)
        if len(df.columns) == 1 or df.empty:
            df = pd.read_csv(data_file, delimiter=",").reset_index(drop=True)
        if (
            df.columns.str.contains("OakFileName").any()
            # and df.columns.str.contains("Include").any()
            # and df.columns.str.contains("PSG").any()
            # and df.columns.str.contains("On Oak").any()
            and df.columns.str.contains("Directory").any()
        ):
            # df = df.loc[(df["Include"] == "Yes") & (df["Type"] == "PSG") & (df["Directory"].notnull())]
            _df = []
            df = df.loc[df["Directory"].notnull()]
            if cohort is not None:
                df = df.query(f"Cohort == '{cohort}' or Cohort == '{cohort.upper()}'").reset_index(drop=True)
            data_list = []
            for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
                directory = Path(row["Directory"])
                # filename = row["PSG file name"]
                filename = row["OakFileName"]
                data_files = list(
                    set(directory.glob(f"{filename}.[eErR][dDeE][fFcC]"))
                    # | set(directory.glob(f"{filename.lower()}*.[eErR][dDeE][fFcC]"))
                    # | set(directory.glob(f"{filename.upper()}*.[eErR][dDeE][fFcC]"))
                    | set(directory.glob(f"BEI {filename[3:]}*.[eErR][dDeE][fFcC]"))
                    # set(directory.rglob(f"{filename}*.[eErR][dDeE][fFcC]"))
                    # | set(directory.rglob(f"{filename.lower()}*.[eErR][dDeE][fFcC]"))
                )
                if len(data_files) == 0:  # FHC upper/lowercase confusion with duplicates
                    data_files = list(set(directory.glob(f"{filename.lower()}*.[eErR][dDeE][fFcC]")))
                    if len(data_files) == 0:
                        data_files = list(set(directory.glob(f"{filename.upper()}*.[eErR][dDeE][fFcC]")))
                if len(data_files) == 0: # Don't know why recursive globbing isn't used above
                    data_files = list(set(directory.rglob(f"{filename}.[EeRr][DdEe][FfCc]")))
                if len(data_files) == 1:
                    if pattern is not None:
                        # data_list = [p for p in data_list if pattern in p.stem]
                        if pattern.lower() in data_files[0].name.lower():
                            data_list.append(data_files[0])
                            _df.append(df.query(f"OakFileName == '{filename}'"))
                    else:
                        data_list.append(data_files[0])
                        df_cand = df.query(f"OakFileName == '{filename}'")
                        if len(df_cand) > 1:
                            print("whaaaa")
                        _df.append(df_cand)
                else:
                    print("BAAAD BOOIIi")

            if _df:
                # df = df.merge(pd.concat(_df))
                df = pd.concat(_df, ignore_index=True)  # Why merge?
        else:
            raise FileNotFoundError("Please supply a .csv file!")

    return data_list, df


def run_inference(args):

    # check_datafile(args.data_file)
    # return
    # Determine data type
    data_paths, df = get_datapaths(
        data_path=args.data_path, data_file=args.data_file, cohort=args.cohort, pattern=args.match_pattern
    )
    # data_paths = data_paths[:10]
    # df = df[:10]

    # Get model and device
    device = torch.device(args.device)
    model = get_model_from_ckpt(ckpt_path=args.model_path, device=device)
    n_channels = model.example_input_array.shape[1]

    # Run over data files
    missing_files = []
    missing_hyp = []
    error_files = []
    success_files = 0
    accuracies = []
    with torch.no_grad():
        logger.info(f"Running inference over {len(data_paths)} files...")
        generator = zip(data_paths, df.iterrows()) if isinstance(df, pd.DataFrame) else zip(data_paths, df)
        with tqdm(generator, total=len(data_paths)) as pbar:
            for i, (data_path, idx_row) in enumerate(pbar):

                if idx_row is not None:
                    idx, row = idx_row
                    cohort = row.Cohort.lower()
                    # subject_id = row["PSG file name"]
                    subject_id = row["OakFileName"]
                else:
                    idx, row = None, None
                    # cohort = data_path.parent.stem.split("_")[0].lower()
                    cohort = args.cohort
                    subject_id = data_path.stem

                # if "060207BB" not in subject_id:
                #     continue
                # if (args.target_dir / cohort / f"preds_{subject_id}.pkl").exists() and (
                #     args.target_dir / cohort / f"preds_{subject_id}.pkl"
                # ).stat().st_size > 1e6:
                #     logger.info(f'Skipping {subject_id}, file exits...')
                #     continue

                if data_path.suffix == ".REC":
                    # data_path = data_path.with_suffix(".edf")
                    filestem = data_path.stem
                    data_path = Path("data") / "cnc" / (filestem + ".edf")

                pbar.set_postfix(cohort=cohort, subject=subject_id)

                # Preprocessing
                channel_map_file = Path("sleep_staging") / "utils" / "channel_dicts" / f"channels_{cohort}.json"
                # if not channel_map_file.exists() and not all([args.eeg, args.eog, args.emg]):
                #     raise FileNotFoundError(f"Channel map file not found: {channel_map_file}, please use appropriate argument flags to designate proper channel names!")
                # elif not channel_map_file.exists() and all([args.eeg, args.eog, args.emg]):
                #     channel_map_file = {'eeg': args.eeg, 'eog': args.eog, 'emg': args.emg}
                try:
                    data, labels, _, stable_sleep, _, _ = process_single_file(
                        str(data_path), args.fs, None, None, cohort, args.encoding, channel_map_file
                    )
                except Exception as err:
                    logger.warning(err)
                    error_files.append(str(data_path))
                    continue

                # Model inference
                N, C, T = data.shape
                data = rearrange(
                    RobustScaler().fit_transform(rearrange(data, "b c t -> (b t) c")),
                    "(b t) c -> b c t",
                    b=N,
                    c=C,
                    t=T,
                )
                if n_channels == 4:
                    data = np.concatenate((data[:, :1], data[:, -3:]), axis=1)
                if labels.shape[-2] / 3600 < 2.5:
                    logger.info(f"Skipping subject due to insufficient data: {labels.shape[-2] / 3600:.2f} hours!")
                    error_files.append(str(data_path))
                    continue
                yhat = model.predict_step(torch.from_numpy(data).to(device))
                yhat["targets"] = labels.squeeze()

                # Calculate accuracy
                acc = metrics.accuracy_score(yhat['targets'].argmax(0)[::30], yhat['yhat_30s'].argmax(1))
                logger.info(f"[ {subject_id} ] Accuracy @ 30 s: {acc:.3f}")
                if len(np.unique(labels)) > 1:
                    accuracies.append(acc)

                # Save outputs
                logger.info(
                    f"[ {subject_id} ] Writing hypnogram labels for {len(np.unique(yhat['targets'].argmax(0)))} classes: {np.unique(yhat['targets'].argmax(0))}"
                )
                if len(np.unique(yhat["targets"].argmax(0))) <= 1:
                    missing_hyp.append(str(data_path))
                cohort_dir = args.target_dir / cohort
                cohort_dir.mkdir(exist_ok=True, parents=True)
                with open(cohort_dir / f"preds_{subject_id}.pkl", "wb") as pkl:
                    pickle.dump(yhat, pkl)
                logger.info(f'[ {subject_id} ] Writing predictions to {cohort_dir / f"preds_{subject_id}.pkl"}')

                success_files += 1

    logger.info(f'Successfully completed {success_files} files!')
    if len(accuracies) >= 3:
        logger.info(f'Accuracy: {np.mean(accuracies):.3f}±{np.std(accuracies):.3f}')
    else:
        logger.info(f'Accuracies: {accuracies}')

    if len(error_files) > 0:
        logger.info(f'Was not able to process {len(error_files)} files:')
        [logger.info(f'\t{f}') for f in error_files]

    if len(missing_files) > 0:
        np.savetxt(f"missing-studies_{args.cohort}_4.txt", missing_files, delimiter="\n", fmt="%s")
    if len(missing_hyp) > 0:
        np.savetxt(f"missing-hypnogram-studies_{args.cohort}.txt", missing_hyp, delimiter="\n", fmt="%s")


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, help="Path to directory containing data files")
    parser.add_argument("--data-file", type=Path, help="Path to .csv file containing data paths")
    parser.add_argument("--match-pattern", type=str, help="Pattern to match in filenames (optional)")
    parser.add_argument("--target-dir", type=Path, required=True, help="Directory to save predictions")
    parser.add_argument("--model-path", type=str, default="trained_models/usleep-large/best_model.ckpt", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Device to run inference on")
    parser.add_argument("--fs", type=int, default=128, help="Sampling frequency of data")
    parser.add_argument("--encoding", type=str, default="raw", help="Data encoding")
    parser.add_argument("--cohort", type=str, default=None, help="Cohort to process")
    # parser.add_argument("--eeg", type=str, default=None, nargs="+", help="Name of EEG channel(s)")
    # parser.add_argument("--eog", type=str, default=None, nargs='+', help="Name of EOG channel(s)")
    # parser.add_argument("--emg", type=str, default=None, nargs='+', help="Name of EMG channel(s)")
    args = parser.parse_args(
        # [
        #     "--data-path",
        #     "data/stages/edf",
        #     "--match-pattern",
        #     "^[^_]+$",  # match all except those with underscores
        #     # "MSQW00001",
        #     "--target-dir",
        #     "tmp",
        #     "--device",
        #     "gpu",
        #     "--cohort",
        #     "stages",
        # ]
    )
    args.device = "cuda" if args.device == "gpu" else args.device

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == len(vars(args)) - 1:
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

    run_inference(args)


if __name__ == "__main__":
    main_cli()
