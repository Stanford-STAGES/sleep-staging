# sleep-staging

This repository contains source code used for training sleep stage detection models presented in Stephansen & Olesen, et *al*. Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy. *Nat Commun* **9**, 5229 (2018). [DOI:10.1038/s41467-018-07229-3](https://doi.org/10.1038/s41467-018-07229-3).

This is a work in progress, and will be updated regularly.

<!-- ## Description

## Installation

### Requirements

-->

## How to run

### Data preparation

#### Create channel mapping file
The data generation pipeline requires at least/most 5/11 distinct channels.
The distinction follows in that only 5 referenced channels are required (1 central EEG, 2 EOG, 1 EMG), but in reality, many EDFs contain unreferenced channels.
The following command ensures that all relevant channels are mapped to a correct category
```
python -m utils.channel_label_identifier -d <data_dir> -o <output_JSON> -c C3 C4 O1 O2 A1 A2 EOGL EOGR EMG EOGRef EMGRef
```
The script will read through the headers of all available EDFs in `data_dir` and then the user will select channels that correspond to a given category given by the `-c` argument.
Shown below is an example of using the command to map the channels from the Cleveland Family Study data obtained from [NSRR](https://sleepdata.org):
```
> python -m utils.channel_label_identifier -d data/cfs/edf -o cfs.json -c C3 C4 O1 O2 A1 A2 EOGL EOGR EMG EOGRef EMGRef
Checking data/cfs/edf for edf files.
Removing any MSLT studies.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 730/730 [02:00<00:00,  6.07it/s]

Enter acceptable channel indices to use for the given identifier.
Use spaces to separate multiple indices.
Total number of EDFs in directory: 730

0.   730 ABDO EFFORT          1.   730 AIRFLOW              2.   730 C3                   3.   730 C4
4.   730 ECG1                 5.   730 ECG2                 6.   730 EMG1                 7.   730 EMG2
8.   730 EMG3                 9.   324 HRate                10.  730 L Leg                11.  730 LOC
12.  323 Light                13.  730 M1                   14.  730 M2                   15.   15 Masimo
16.  729 NASAL PRES           17.  730 OX STATUS            18.    1 PAP FLOW             19.  730 POSITION
20.  730 PULSE                21.  324 PlethWV              22.  730 R Leg                23.  730 ROC
24.  730 SNORE                25.  375 SUM                  26.  406 SaO2                 27.  324 SpO2
28.  730 THOR EFFORT

C3: 2
Selected:  ['C3']
C4: 3
Selected:  ['C4']
O1:
Selected:  []
O2:
Selected:  []
A1: 13
Selected:  ['M1']
A2: 14
Selected:  ['M2']
EOGL: 11
Selected:  ['LOC']
EOGR: 23
Selected:  ['ROC']
EMG: 7 8
Selected:  ['EMG2', 'EMG3']
EOGRef: 14 13
Selected:  ['M2', 'M1']
EMGRef: 6
Selected:  ['EMG1']
```
The script will output the name and count of each channel label available in the data directory, and the user will then use numerical values to choose the correct channel mappings for each category.
Notice that a category can contain multiple indices, which are space-separated and ordered (highest priority channel name should be the first element, and so on).

#### EDF loading routines
The channel mapping `JSON` must be linked to the data cohort identifier by adding a function in `utils/edf_utils.py` that loads the relevant channels given the correct mapping defined above.
The following template function is supplied in `edf_utils.py`:
```python
def load_edf_template(filepath, fs):

    with open("path/to/mychannelmapping.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)
```
The user is responsible for adding a custom `load_edf_X` function based on the template, where `<path/to/mychannelmapping.json>` has been changed to the correct path.
When the function has been added, add a reference to the function in the `edf_read_fns` dictionary in `utils/__init__.py`.

<!--
### Training

### Testing

## Citation

## Contributing

## License -->

<!-- # Description

This repository represents the sleep staging classification work down using neural networks at Stanford University, and is intended primarily for research and historical reference.

Those interested in using the sleep staging classification methods that were developed from this should use the primary, [Stanford-STAGES](https://www.github.com/stanford-stages/stanford-stages) repository.

# sleep-staging


# sc_train.py is run by adding an option with the following format:
Example:
python sc_train.py --model ac_lh_ls_lstm

The ac specifies the CC model configuration, the lh specifies the complexity - high in this case, the ls specifies the window length - 15 seconds in this case, and lstm specifies that the model has memory.

To train a model, the sc_config.py should be changed to match the destination for training files, and similarly, to test a model (which has the same option as training) the destination for testing files should be changed. -->
