import os
import json

import numpy as np
import mne

from utils.errors import MissingHypnogramError
from utils.errors import MissingSignalsError
from utils.errors import ReferencingError


def load_edf(filepath, fs, channel_dict):

    header = mne.io.read_raw_edf(filepath, verbose=False)
    available_channels = header.ch_names

    # Get indices of loaded channels
    channels_to_load = {k: None for k in channel_dict["categories"]}
    for k in channels_to_load.keys():
        for v in channel_dict[k]:
            try:
                channels_to_load[k] = available_channels.index(v)
            except:
                pass

    # Test for missing signals (we assume unreferenced and referenced channels are in the same category, presence of the reference channels determine resampling.)
    missing_central = all([channels_to_load["C3"] is None, channels_to_load["C4"] is None])
    missing_occipital = all([channels_to_load["O1"] is None, channels_to_load["O2"] is None])
    missing_eog = all([channels_to_load["EOGL"] is None, channels_to_load["EOGR"] is None])
    missing_emg = all([channels_to_load["EMG"] is None])
    # if any([missing_central, missing_occipital, missing_eog, missing_emg]):
    #     raise MissingSignalsError(os.path.basename(filepath), available_channels)

    # Preload
    data = [[]] * len(channel_dict["categories"])
    sampling_rates = [[]] * len(channel_dict["categories"])
    for i, k in enumerate(channel_dict["categories"]):
        if channels_to_load[k] is not None:
            data[i] = mne.io.read_raw_edf(
                filepath,
                verbose=False,
                preload=False,
                exclude=[ch for j, ch in enumerate(available_channels) if j != channels_to_load[k]],
            )
            sampling_rates[i] = int(data[i].info["sfreq"])
            if len(data[i].ch_names) > 1:
                data[i] = data[i][data[i].ch_names.index(available_channels[channels_to_load[k]])]

    # Reference data
    if data[7]:  # A1
        if data[1]:
            data[1] = (data[1][:][0] - data[7][:][0]).squeeze()  # C4 to A1
        if data[3]:
            data[3] = (data[3][:][0] - data[7][:][0]).squeeze()  # O2 to A1
    else:
        if data[1]:
            data[1] = data[1][:][0].squeeze()
        if data[3]:
            data[3] = data[3][:][0].squeeze()
    if data[8]:  # A2
        if data[0]:
            data[0] = (data[0][:][0] - data[8][:][0]).squeeze()  # C3 to A2
        if data[2]:
            data[2] = (data[2][:][0] - data[8][:][0]).squeeze()  # O1 to A2
    else:
        if data[0]:
            data[0] = data[0][:][0].squeeze()
        if data[2]:
            data[2] = data[2][:][0].squeeze()
    if data[9]:  # EOG Ref (most cases A2)
        try:
            if data[4]:
                data[4] = (data[4][:][0] - data[9][:][0]).squeeze()  # EOGL to EOG Ref
            if data[5]:
                data[5] = (data[5][:][0] - data[9][:][0]).squeeze()  # EOGR to EOG Ref
        except:
            if data[4]:
                data[4] = data[4][:][0].squeeze()
            if data[5]:
                data[5] = data[5][:][0].squeeze()
    else:
        if data[4]:
            data[4] = data[4][:][0].squeeze()
        if data[5]:
            data[5] = data[5][:][0].squeeze()
    if data[10]:  # EMG Ref
        if data[6]:
            data[6] = (data[6][:][0] - data[10][:][0]).squeeze()  # EMG to EMG Ref
    else:
        if data[6]:
            data[6] = data[6][:][0].squeeze()
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

    with open(channel_map_file) as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)
