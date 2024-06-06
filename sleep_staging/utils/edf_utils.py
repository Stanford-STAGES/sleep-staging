import logging
import os
import json
import warnings

import numpy as np
import mne

from sleep_staging.utils.errors import MissingHypnogramError
from sleep_staging.utils.errors import MissingSignalsError
from sleep_staging.utils.errors import ReferencingError
from sleep_staging.utils.channel_label_identifier import run_channel_label_identifier


UNIT_SCALING = {"µV": 1, "mV": 1e3, "V": 1e6}


def load_edf(filepath, fs, channel_dict):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        header = mne.io.read_raw_edf(filepath, verbose=False)
        available_channels = header.ch_names

    # Get indices of loaded channels
    channels_to_load = {k: None for k in channel_dict["categories"]}
    for k in channels_to_load.keys():
        for v in channel_dict[k]:
            try:
                channels_to_load[k] = available_channels.index(v)
                break
            except:
                pass
    reference_eeg = True
    reference_eog = True
    reference_emg = True
    ref_channels = [header.ch_names[j] for j in [channels_to_load[k] for k in ["A1", "A2", "EMGRef"]] if j is not None]
    eog_channels = [header.ch_names[j] for j in [channels_to_load[k] for k in ["EOGL", "EOGR"]] if j is not None]
    eeg_channels = [
        header.ch_names[j] for j in [channels_to_load[k] for k in ["C3", "C4", "O1", "O2"]] if j is not None
    ]
    emg_channels = [header.ch_names[j] for j in [channels_to_load[k] for k in ["EMG"]] if j is not None]
    if ref_channels:
        if any([ref in ch for ref in ref_channels for ch in eeg_channels]):
            reference_eeg = False
        if any([ref in ch for ref in ref_channels for ch in eog_channels]):
            reference_eog = False
        if any([ref in ch for ref in ref_channels for ch in emg_channels]):
            reference_emg = False
    else:
        reference_eeg, reference_eog, reference_emg = False, False, False

    # Get the physical units (used for rescaling inputs, some LOC/ROC channels have been found in mV while references are in muV)
    signal_units = {}
    for i, k in enumerate(channel_dict["categories"]):
        for j, ch in enumerate(available_channels):
            if j == channels_to_load[k]:
                signal_units[k] = header._orig_units[ch]

    # Test for missing signals (we assume unreferenced and referenced channels are in the same category, presence of the reference channels determine resampling.)
    missing_central = all([channels_to_load["C3"] is None, channels_to_load["C4"] is None])
    missing_occipital = all([channels_to_load["O1"] is None, channels_to_load["O2"] is None])
    missing_eog = all([channels_to_load["EOGL"] is None, channels_to_load["EOGR"] is None])
    missing_emg = all([channels_to_load["EMG"] is None])
    if any([missing_central, missing_eog, missing_emg]):
        raise MissingSignalsError(os.path.basename(filepath), available_channels)

    # Preload
    data = [[]] * len(channel_dict["categories"])
    sampling_rates = [[]] * len(channel_dict["categories"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, k in enumerate(channel_dict["categories"]):
            if channels_to_load[k] is not None:
                data[i] = mne.io.read_raw_edf(
                    filepath,
                    verbose=False,
                    preload=False,
                    exclude=[ch for j, ch in enumerate(available_channels) if j != channels_to_load[k]],
                )
                sampling_rates[i] = int(data[i].info["sfreq"])
                if len(data[i].ch_names) > 1:  # TODO: this should be merged with the below block
                    if header._orig_units[available_channels[channels_to_load[k]]] != "n/a":
                        logging.debug(
                            f"Scaling {available_channels[channels_to_load[k]]} with {header._orig_units[available_channels[channels_to_load[k]]]}: {UNIT_SCALING[header._orig_units[available_channels[channels_to_load[k]]]]}"
                        )
                        data[i] = (
                            UNIT_SCALING[header._orig_units[available_channels[channels_to_load[k]]]]
                            * data[i].pick(available_channels[channels_to_load[k]]).get_data()
                        )
                    else:
                        data[i] = data[i].pick(available_channels[channels_to_load[k]]).get_data()
                else:
                    if header._orig_units[available_channels[channels_to_load[k]]] != "n/a":
                        logging.debug(
                            f"Scaling {available_channels[channels_to_load[k]]} with {header._orig_units[available_channels[channels_to_load[k]]]}: {UNIT_SCALING[header._orig_units[available_channels[channels_to_load[k]]]]}"
                        )
                        data[i] = (
                            UNIT_SCALING[header._orig_units[available_channels[channels_to_load[k]]]]
                            * data[i].pick(available_channels[channels_to_load[k]]).get_data()
                        )
                    else:
                        data[i] = data[i].pick(available_channels[channels_to_load[k]]).get_data()

    # Reference data
    if isinstance(data[7], np.ndarray) and reference_eeg:  # A1
        if isinstance(data[1], np.ndarray):
            data[1] = (data[1] - data[7]).squeeze()  # C4 to A1
        if isinstance(data[3], np.ndarray):
            data[3] = (data[3] - data[7]).squeeze()  # O2 to A1
    else:
        if isinstance(data[1], np.ndarray):
            data[1] = data[1].squeeze()
        if isinstance(data[3], np.ndarray):
            data[3] = data[3].squeeze()
    if isinstance(data[8], np.ndarray) and reference_eeg:  # A2
        if isinstance(data[0], np.ndarray):
            data[0] = (data[0] - data[8]).squeeze()  # C3 to A2
        if isinstance(data[2], np.ndarray):
            data[2] = (data[2] - data[8]).squeeze()  # O1 to A2
    else:
        if isinstance(data[0], np.ndarray):
            data[0] = data[0].squeeze()
        if isinstance(data[2], np.ndarray):
            data[2] = data[2].squeeze()
    if isinstance(data[9], np.ndarray) and reference_eog:  # EOG Ref (most cases A2)
        try:
            if isinstance(data[4], np.ndarray):
                data[4] = (data[4] - data[9]).squeeze()  # EOGL to EOG Ref
            if isinstance(data[5], np.ndarray):
                data[5] = (data[5] - data[9]).squeeze()  # EOGR to EOG Ref
        except:
            if isinstance(data[4], np.ndarray):
                data[4] = data[4].squeeze()
            if isinstance(data[5], np.ndarray):
                data[5] = data[5].squeeze()
    else:
        if isinstance(data[4], np.ndarray):
            data[4] = data[4].squeeze()
        if isinstance(data[5], np.ndarray):
            data[5] = data[5].squeeze()
    if isinstance(data[10], np.ndarray) and reference_emg:  # EMG Ref
        if isinstance(data[6], np.ndarray):
            data[6] = (data[6] - data[10]).squeeze()  # EMG to EMG Ref
    else:
        if isinstance(data[6], np.ndarray):
            data[6] = data[6].squeeze()

    # # Reference data
    # if data[7] and reference_eeg:  # A1
    #     if data[1]:
    #         data[1] = (data[1][:][0] - data[7][:][0]).squeeze()  # C4 to A1
    #     if data[3]:
    #         data[3] = (data[3][:][0] - data[7][:][0]).squeeze()  # O2 to A1
    # else:
    #     if data[1]:
    #         data[1] = data[1][:][0].squeeze()
    #     if data[3]:
    #         data[3] = data[3][:][0].squeeze()
    # if data[8] and reference_eeg:  # A2
    #     if data[0]:
    #         data[0] = (data[0][:][0] - data[8][:][0]).squeeze()  # C3 to A2
    #     if data[2]:
    #         data[2] = (data[2][:][0] - data[8][:][0]).squeeze()  # O1 to A2
    # else:
    #     if data[0]:
    #         data[0] = data[0][:][0].squeeze()
    #     if data[2]:
    #         data[2] = data[2][:][0].squeeze()
    # if data[9] and reference_eog:  # EOG Ref (most cases A2)
    #     try:
    #         if data[4]:
    #             data[4] = (data[4][:][0] - data[9][:][0]).squeeze()  # EOGL to EOG Ref
    #         if data[5]:
    #             data[5] = (data[5][:][0] - data[9][:][0]).squeeze()  # EOGR to EOG Ref
    #     except:
    #         if data[4]:
    #             data[4] = data[4][:][0].squeeze()
    #         if data[5]:
    #             data[5] = data[5][:][0].squeeze()
    # else:
    #     if data[4]:
    #         data[4] = data[4][:][0].squeeze()
    #     if data[5]:
    #         data[5] = data[5][:][0].squeeze()
    # if data[10] and reference_emg:  # EMG Ref
    #     if data[6]:
    #         data[6] = (data[6][:][0] - data[10][:][0]).squeeze()  # EMG to EMG Ref
    # else:
    #     if data[6]:
    #         data[6] = data[6][:][0].squeeze()
    del data[7:]
    del sampling_rates[7:]

    return data, sampling_rates[:7], channel_dict["categories"][:7]


def load_edf_ahc(filepath, fs):

    with open("utils/channel_dicts/channels_ahc.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)


def load_edf_dhc(filepath, fs):

    with open("utils/channel_dicts/channels_dhc.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)


def load_edf_ihc(filepath, fs):

    with open("utils/channel_dicts/channels_ihc.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)


def load_edf_khc(filepath, fs):

    with open("utils/channel_dicts/channels_khc.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)


def load_edf_jcts(filepath, fs):

    with open("utils/channel_dicts/channels_jcts.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)


def load_edf_wsc(filepath, fs):

    with open("utils/channel_dicts/channels_wsc.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)


def load_edf_cfs(filepath, fs):

    with open("utils/channel_dicts/channels_cfs.json") as json_file:
        channel_dict = json.load(json_file)
        channel_dict.pop("F3", None)
        channel_dict.pop("F4", None)
        channel_dict["categories"].remove("F3")
        channel_dict["categories"].remove("F4")

    return load_edf(filepath, fs, channel_dict)


def load_edf_chat(filepath, fs):

    with open("utils/channel_dicts/channels_chat.json") as json_file:
        channel_dict = json.load(json_file)
        channel_dict.pop("F3", None)
        channel_dict.pop("F4", None)
        channel_dict["categories"].remove("F3")
        channel_dict["categories"].remove("F4")

    return load_edf(filepath, fs, channel_dict)


def load_edf_mesa(filepath, fs):

    with open("utils/channel_dicts/channels_mesa.json") as json_file:
        channel_dict = json.load(json_file)
        channel_dict.pop("F3", None)
        channel_dict.pop("F4", None)
        channel_dict["categories"].remove("F3")
        channel_dict["categories"].remove("F4")

    return load_edf(filepath, fs, channel_dict)


def load_edf_mros(filepath, fs):

    with open("utils/channel_dicts/channels_mros.json") as json_file:
        channel_dict = json.load(json_file)
        channel_dict.pop("F3", None)
        channel_dict.pop("F4", None)
        channel_dict["categories"].remove("F3")
        channel_dict["categories"].remove("F4")

    return load_edf(filepath, fs, channel_dict)


def load_edf_shhs(filepath, fs):

    with open("utils/channel_dicts/channels_shhs.json") as json_file:
        channel_dict = json.load(json_file)
        channel_dict.pop("F3", None)
        channel_dict.pop("F4", None)
        channel_dict["categories"].remove("F3")
        channel_dict["categories"].remove("F4")

    return load_edf(filepath, fs, channel_dict)


def load_edf_ssc(filepath, fs):

    with open("utils/channel_dicts/channels_ssc.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)


def load_edf_dcsm(filepath, fs):

    with open("utils/channel_dicts/dcsm.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)


def load_edf_template(filepath, fs):

    with open("path/to/mychannelmapping.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)


def load_edf_mapfile(filepath, fs, channel_map_file):

    # if isinstance(channel_map_file, dict):
    #     channel_dict = channel_map_file
    # else:
    if channel_map_file.exists():
        with open(channel_map_file) as json_file:
            channel_dict = json.load(json_file)
    else:
        # If the file is not found, we assume the EDF file is in the same directory as the other files
        if filepath[-4:] == ".edf":
            filepath = filepath.split('/')
        channel_dict = run_channel_label_identifier("/".join(filepath[:-1]), channel_map_file, ["C3", "C4", "O1", "O2", "EOGL", "EOGR", "EMG", "A1", "A2", "EOGRef", "EMGRef"])
        filepath = "/".join(filepath)

    return load_edf(filepath, fs, channel_dict)
