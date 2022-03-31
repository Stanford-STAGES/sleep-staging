# sleep-staging

This repository contains source code used for training sleep stage detection models presented in Stephansen & Olesen, et *al*. Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy. *Nat Commun* **9**, 5229 (2018). [DOI:10.1038/s41467-018-07229-3](https://doi.org/10.1038/s41467-018-07229-3).

This is a work in progress, and will be updated regularly.

## Table of contents
1. [Requirements](#requirements)
2. [How to run](#how-to-run)
3. [Example run](#detailed-example)

## Requirements
The necessary packages can be installed in a `conda` environment by running the following command from the root directory.
```
conda env create -f environment.yml
```
*Note: the installation process may take a couple of minutes*

## How to run

### Data preparation

#### Create channel mapping file
The data generation pipeline requires at least/most 5/11 distinct channels.
The distinction follows in that only 5 referenced channels are required (1 central EEG, 2 EOG, 1 EMG), but in reality, many EDFs contain unreferenced channels.
The following command ensures that all relevant channels are mapped to a correct category
```
python -m utils.channel_label_identifier -d <data_dir> \
                                         -o <output_JSON> \
                                         -c C3 C4 O1 O2 EOGL EOGR EMG A1 A2 EOGRef EMGRef
```
The script will read through the headers of all available EDFs in `data_dir` and then the user will select channels that correspond to a given category given by the `-c` argument.
Shown below is an example of using the command to map the channels from the Cleveland Family Study data obtained from [NSRR](https://sleepdata.org):
```
> python -m utils.channel_label_identifier -d data/cfs/edf -o cfs.json -c C3 C4 O1 O2 EOGL EOGR EMG A1 A2 EOGRef EMGRef
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

#### Hypnogram loading routines
Similar to the EDF loading, the user is also responsible for ensuring correct loading of hypnogram files.
A `load_hypnogram_default()` function is supplied in `utils/sta_utils.py` (pending namechange).
This can be used as is, or modified to custom filetypes.
Importantly, any custom hypnogram loading routine should return a Numpy array of shape `(N,)` containing integers mapping from `{Wake, N1, N2, N3, REM[, MOVEMENT/UNKNOWN]}` to `{1, 2, 3, 4, 5[, 7]}` or from `{Wake, S1, S2, S3, S4, REM[, MOVEMENT/UNKNOWN]}` to `{1, 2, 3, 4, 4, 5[, 7]}`.

#### Run data preprocessing

### Training
To train a new model, run the following command from the root directory (`~/sleep-staging`):
```
python train.py [OPTIONS]
```
The `[OPTIONS]` can be a list of input arguments controlling various aspects of the model training, such as the datamodule, model architecture, optimizer, etc. as well as all the flags listed in the [PyTorch Lightning Trainer API](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).
The full list of optional flags can be shown by running `python train.py --help`

## Detailed example
The following example will go through the steps of 1) downloading and 2) preparing a dataset, 3) train a model on the dataset, and finally 4) run evaluation.
We will use the Danish Center for Sleep Medicine Cohort (DCSM, presented [here][usleep]), which contains 255 PSGs from a diverse group of patients with sleep disorders.

### Dataset preparation
Download the DSCM dataset by calling the `fetch_data` routine:
```
python -m preprocessing.download_data.fetch_data -d dcsm \
                                                 -o data/dcsm/edf \
                                                 -n 20
```
This will place the first 20 recordings of the dataset in the `data/dcsm` folder, but you can change this to accomodate your own preferences.

Run the channel mapping routine using the following command:
```
python -m utils.channel_label_identifier -d data/dcsm/edf \
                                         -o utils/channel_dicts/dcsm.json \
                                         -c C3 C4 O1 O2 EOGL EOGR EMG A1 A2 EOGRef EMGRef
```
<!-- Before running the preprocessing pipeline, we add the following function in `utils/edf_utils.py` in order to use the correct channel mapping JSON
(we also remember to add the `{"dcsm": load_edf_dcsm}` key-val pair to the `edf_read_fns` in `utils/__init__.py`):
```
def load_edf_dcsm(filepath, fs):

    with open("utils/channel_dicts/dcsm.json") as json_file:
        channel_dict = json.load(json_file)

    return load_edf(filepath, fs, channel_dict)
``` -->
The following function is added to `utils/sta_utils.py` to load correct hypnograms (DCSM hypnograms are in .ids format (Index/Duration/Stage)):
```
def load_hypnogram_ids(hyp_file):

    parts = hyp_file.split(".")
    if len(parts) > 1:
        parts = parts[0]
    dirname, basename = os.path.split(hyp_file)
    if basename == "hypnogram":
        hyp_file = os.path.join(dirname, basename + ".ids")
    else:
        hyp_file = os.path.join(dirname, "hypnogram.ids")

    df = pd.read_csv(hyp_file, header=None)
    dur = df[1].values // 30
    stages = df[2].values
    hypnogram = [STAGE_MAP[s] for (d, s) in zip(dur, stages) for _ in range(d)]

    return np.asarray(hypnogram)[:, np.newaxis]
```
We remember to add `{"dcsm": load_hypnogram_ids}` to `hypnogram_read_fns` in `utils/sta_utils.py`.
The preprocessing pipeline which resamples, filters, and segments the PSG data can finally run by the following command:
```
python -m preprocessing.process_data -d data/dcsm/edf \
                                     -o data/dcsm/raw \
                                     --encoding raw \
                                     --seq_len 10 \
                                     --overlap 5 \
                                     --fs 128 \
                                     --channel_map_file utils/channel_dicts/dcsm.json \
                                     -c dcsm \
```
This will place the H5 files in the `data/dcsm/raw` folder corresponding to the type of encoding.

### Run training procedure
The following command will train a sleep stage model using the `massc_average` architecture on the DSCM data:
```
python train.py --data_dir data/dcsm/raw \
                --model_type massc_average \
                --gpus 1 \
                --block_type simple \
                --balanced_sampling \
                --n_records 20 \
                --scaling robust \
                --name dcsm \
                --batch_size 32
```
#### Using multiple threads for dataloading
By default, the datamodule will use only the main process for the train and eval data loaders.
If more cores are available, training can be sped up by using the `--n_workers N` flag in the training command above, while setting `N > 0`.

#### Multi-GPU training
As mentioned, all the flags from PyTorch Lightning are available here, so in order to do multi-GPU training, one has simply to add `--gpus N` and `--accelerator ddp` to the flags supplied to `train.py`.

### Get predictions
After training a model, predictions can be acquired using the following commmand
```
python predict.py --resume_from_checkpoint experiments/dcsm
```
By default, this will only return predictions for the validation data used in the training phase.
These are placed in the `experiments/dcsm/predictions/` folder.

#### Predicting on new data
If a new dataset is acquired, the user can get predictions for this by running the data preprocessing pipeline above and passing a cohort-path key-val pair to the `predict` command:
```
python predict.py --resume_from_checkpoint experiments/dcsm \
                  --predict_on '{"<cohort_name>": "<path_to_cohort>", "<another_cohort_name>": "<path_to_another_cohort>"}'
```
*Note the placement of pings and double pings.*

Similarly to above, prediction speed is affected by adding a GPU and adding more workers by using the `--gpus` and `--n_workers` flags, respectively.

[usleep]: https://doi.org/10.1038/s41746-021-00440-5

<!-- #### Example training run -->
<!--
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
