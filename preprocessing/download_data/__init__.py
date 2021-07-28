from .dcsm import download_dcsm
from .sedfx import download_sedfx_sc, download_sedfx_st


download_fns = {
    "dcsm": download_dcsm,
    "sedfx_sc": download_sedfx_sc,
    "sedfx_st": download_sedfx_st
    # "physionet": None,
}


def download_dataset(dataset_name, out_dir, n_first):
    download_fns[dataset_name](out_dir, n_first)
