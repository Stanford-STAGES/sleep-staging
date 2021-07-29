# @requirements:
# pip3 install --upgrade pip
# pip3 install pyedflib
# pip3 install -U wxPython
# @description
# Tool for (1) identifying all unique .edf channel names in a specified directory and
# (2) creating categories to group like channels for processing in a pipeline.
# Creates json file for with fields named by the categories provided as well as
# - pathname: Path of directory initially parsed
# - edfFiles: List of all edf files parsed for channel names
# - categories: List of categories (string labels) provided, which match the remaining field names
# @usage From the command line:
#  python channel_label_identifier.py <path name with .edf files> [channel_categories, separated_by_spaces]
#
# @example Create a json file with categories for C3 and C4 through user selection:
#   python channel_label_identifier . C3 C4
# @example List all unique signal labels found in the current (.) directory
# python channel_label_identifier .
#
# @author Hyatt Moore
# @date 2/20/2018

import argparse
import json
import os
from glob import glob
import sys
from collections import Counter
from pathlib import Path

from joblib import delayed

from utils import ParallelExecutor

try:
    import mne
except:
    from pyedflib import EdfReader
from tqdm import tqdm

# from pyedflib import EdfReader

JSON_FILENAME = "signal_labels.json"


# Wrapper for getEDFFiles
def getEDFFilenames(path2check):
    edfFiles = getEDFFiles(path2check)
    return [str(i) for i in edfFiles]


def getEDFFiles(path2check):
    p = Path(path2check)
    # verify that we have an accurate directory
    # if so then list all .edf/.EDF files
    # if p.is_dir():
    if os.path.isdir(path2check):
        print("Checking", path2check, "for edf files.")
        # edfFiles = list(p.rglob("*.[EeRr][DdEe][FfCc]"))  # make search case-insensitive
        edfFiles = glob(os.path.join(path2check, "*.[EeRr][DdEe][FfCc]"))
        print("Removing any MSLT studies.")
        # edfFiles = [edf for edf in edfFiles if not "mslt" in edf.stem.lower()]
        edfFiles = [edf for edf in edfFiles if "mslt" not in os.path.basename(edf.lower())]
    else:
        print(path2check, " is not a valid directory.")
        edfFiles = []
    return edfFiles


def getSignalHeaders(edfFilename):
    try:
        # print("Reading headers from ", edfFilename)
        try:
            edfR = EdfReader(str(edfFilename))
            return edfR.getSignalHeaders()
        except:
            edfR = mne.io.read_raw_edf(str(edfFilename), verbose=False)
            return edfR.ch_names
    except:
        print("Could not read headers from {}".format(edfFilename))
        return []


def getChannelLabels(edfFilename):
    channelHeaders = getSignalHeaders(edfFilename)
    try:
        return [fields["label"] for fields in channelHeaders]
    except:
        return channelHeaders


def displaySetSelection(label_set):
    numCols = 4
    curItem = 0
    width = 30
    rowStr = ""
    for label, count in sorted(label_set.items()):
        rowStr += (f"{curItem}.".ljust(4) + f"{count}".rjust(4).ljust(5) + f"{label}").ljust(width)
        # rowStr = rowStr + str(str(str(curItem) + ".").ljust(4) + f"{count}".rjust(5) + f"{label}").ljust(
        #     width
        # )
        curItem = curItem + 1
        if curItem % numCols == 0:
            print(rowStr)
            rowStr = ""
    if len(rowStr) > 0:
        print(rowStr)


def getAllChannelLabels(path2check):
    edfFiles = getEDFFilenames(path2check)
    num_edfs = len(edfFiles)
    if num_edfs == 0:
        label_list = []
    else:
        label_set = getLabelSet(edfFiles)
        label_list = sorted(label_set)
    return label_set, num_edfs


def getAllChannelLabelsWithCounts(edfFiles):
    num_edfs = len(edfFiles)
    if num_edfs == 0:
        label_list = []
    else:

        def func(fp):
            return getChannelLabels(fp)

        output = ParallelExecutor(n_jobs=-1, prefer="threads")(total=len(edfFiles))(
            delayed(func)(edfFile) for edfFile in edfFiles
        )
        label_set_counts = Counter([l2 for l1 in output for l2 in l1])
        # label_list = []
        # for edfFile in tqdm(edfFiles):
        #     [label_list.append(l) for l in getChannelLabels(edfFile)]
        # label_set_counts = Counter(label_list)
    return label_set_counts, num_edfs


def getLabelSet(edfFiles):
    label_set = set()
    for edfFile in edfFiles:
        # only add unique channel labels to our set`
        label_set = label_set.union(set(getChannelLabels(edfFile)))
    return label_set


def printUsage(toolName):
    print("Usage:\n\t", toolName, " <pathname to search> <channel category> {<channel category>}")
    print("Example:\n\t", toolName, " . C3 C4")


def run(args):
    path2check = args.data_dir
    json_filename = args.json_out
    # jsonFileOut = json_filename
    # jsonFileOut = Path('./src/config/signal_labels').joinpath(json_filename)
    # jsonFileOut = Path(path2check).joinpath(json_filename)
    # jsonFileOut = Path('/home/alexno/Documents/utils').joinpath(JSON_FILENAME)
    # jsonFileOut = Path(path2check).joinpath(JSON_FILENAME)
    channelsToID = args.channels

    edfFiles = getEDFFilenames(path2check)
    num_edfs = len(edfFiles)
    if num_edfs == 0:
        print("No file(s) found!")
    else:
        label_set_counts, _ = getAllChannelLabelsWithCounts(edfFiles)
        # print(label_set_counts)
        # label_set = getLabelSet(edfFiles)
        label_list = sorted(list(label_set_counts.keys()))
        # label_list = sorted(label_set)
        print()

        if len(channelsToID) > 0:
            print(
                "Enter acceptable channel indices to use for the given identifier. \n"
                "Use spaces to separate multiple indices. \n"
                f"Total number of EDFs in directory: {num_edfs}"
            )
            print()

        displaySetSelection(label_set_counts)
        print()

        if len(channelsToID) > 0:

            toFile = {}  # dict()
            toFile["pathname"] = path2check  # a string
            toFile["edfFiles"] = edfFiles  # a list
            toFile["categories"] = channelsToID  # a list of strings

            for ch in channelsToID:
                indices = [int(num) for num in input(ch + ": ").split()]
                selectedLabels = [label_list[i] for i in indices]
                print("Selected: ", selectedLabels)
                toFile[ch] = selectedLabels

            with open(json_filename, "w") as json_file:
                json.dump(toFile, json_file, indent=4, sort_keys=True)
            # jsonStr = json.dumps(toFile, indent=4, sort_keys=True)
            # json_filename.write_text(jsonStr)
            print(json.dumps(toFile))
            print()
            print("JSON data written to file:", json_filename)


if __name__ == "__main__":
    # if number of input arguments is none
    # usage_str = "Usage:\n\t", sys.argv[0], " <pathname to search> < <channel category> {<channel category>}")
    # print("Example:\n\t", toolName, " . C3 C4"
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", required=True, type=str, help="Location of EDF(s) to check")
    parser.add_argument(
        "-o", "--json_out", required=True, type=str, help="Location of the output JSON file containing channel mappings"
    )
    parser.add_argument("-c", "--channels", required=True, nargs="+", help="List of channels to map")
    args = parser.parse_args()
    if len(sys.argv) < 2:
        printUsage(sys.argv[0])
    else:
        run(args)
