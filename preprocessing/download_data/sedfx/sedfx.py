import os
from ..utils import download_dataset

# Get path to current module file
_FILE_PATH = os.path.split(__file__)[0]

# SEDF-SC globals
_SERVER_URL_SC = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
_CHECKSUM_FILE_SC = f"{_FILE_PATH}/sedfx_sc_checksums.txt"

# SEDF-ST globals
_SERVER_URL_ST = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry"
_CHECKSUM_FILE_ST = f"{_FILE_PATH}/sedfx_st_checksums.txt"


def sedf_paths_func(file_name, server_url, out_dataset_folder):
    """
    See utime/preprocessing/dataset_preparation/utils.py [download_dataset]
    A callable of signature func(file_name, server_url, out_dataset_folder) which returns:
    1) download_url (path to fetch file from on remote system)
    2) out_file_path (path to store file on local system)
    """
    out_subject_folder = file_name.split("-")[0][:-1] + "0"
    out_subject_folder = os.path.join(out_dataset_folder, out_subject_folder)
    out_file_path = os.path.join(out_subject_folder, file_name)
    download_url = server_url + f"/{file_name}"
    return download_url, out_file_path


def download_sedfx_sc(out_dataset_folder, N_first=None):
    """ Download the Sleep-EDF sleep-cassette (153 records) dataset """
    download_dataset(
        out_dataset_folder=out_dataset_folder,
        server_url=_SERVER_URL_SC,
        checksums_path=_CHECKSUM_FILE_SC,
        paths_func=sedf_paths_func,
        N_first=N_first * 2 if N_first else None,  # Two items per subject
    )


def download_sedfx_st(out_dataset_folder, N_first=None):
    """ Download the Sleep-EDF sleep-telemetry (44 records) dataset """
    download_dataset(
        out_dataset_folder=out_dataset_folder,
        server_url=_SERVER_URL_ST,
        checksums_path=_CHECKSUM_FILE_ST,
        paths_func=sedf_paths_func,
        N_first=N_first * 2 if N_first else None,  # Two items per subject
    )
