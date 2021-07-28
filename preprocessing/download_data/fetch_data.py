import argparse
import os

from preprocessing.download_data import download_dataset, download_fns

available_datasets = set(download_fns.keys())


def fetch_data(args):

    # Make directory
    if not os.path.exists(args.output_dir):
        print("Creating output directory {args.out_dir}")
        os.makedirs(args.output_dir)

    # Actually download data
    download_dataset(args.dataset, args.output_dir, args.n_first)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset to download.", choices=available_datasets)
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to output directory.\nWill be created if not available."
    )
    parser.add_argument("-n", "--n_first", help="Number of recordings to download.")
    args = parser.parse_args()
    fetch_data(args)
