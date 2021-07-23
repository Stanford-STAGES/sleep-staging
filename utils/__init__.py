from .arg_utils import get_args
from .chunking import chunks
from .dataset_utils import get_data
from .edf_utils import *
from .evaluate_performance import (
    evaluate_performance,
    transition_matrix,
)
from .h5_utils import (
    get_h5_info,
    load_h5_data,
    load_psg_h5_data,
    initialize_record,
    get_class_sequence_idx,
    get_stable_sleep_periods,
    get_stable_stage,
    get_unknown_stage,
    SCALERS,
)
from .logger_callback_utils import get_loggers_callbacks
from .multitaper_spectrogram import multitaper_spectrogram
from .model_utils import get_model
from .parallel_bar import ParallelExecutor
from .plotting import (
    extended_epoch_view,
    plot_data,
    plot_psg_hypnogram_hypnodensity,
    plot_segment,
    view_record,
)
from .sta_utils import load_scored_data

read_fns = {
    "cc": load_h5_data,
    "raw": load_psg_h5_data,
}

edf_read_fns = {
    "ahc": load_edf_ahc,
    "dhc": load_edf_dhc,
    "wsc": load_edf_wsc,
    "ihc": load_edf_ihc,
    "khc": load_edf_khc,
    "jcts": load_edf_jcts,
    "cfs": load_edf_cfs,
    "chat": load_edf_chat,
    "mesa": load_edf_mesa,
    "mros": load_edf_mros,
    "shhs": load_edf_shhs,
    "ssc": load_edf_ssc,
}

# __all__ = ["evaluate_performance", "load_h5_data", "ParallelExecutor", "read_fns", "get_args", "get_model"]
