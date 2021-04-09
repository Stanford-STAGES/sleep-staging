import argparse
import json
import logging
import os
import pickle
import tempfile
os.chdir(os.path.expandvars('$HOME/sleep-staging'))
print(os.getcwd())

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from h5py import File
from tqdm import tqdm

try:
    from ..utils.plotting import view_record, extended_epoch_view
except:
    from utils.plotting import view_record, extended_epoch_view

sns.set(context='paper', rc={'figure.figsize':(11,10)})

HYP_DICT = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}
logging.basicConfig(format='%(asctime)s %(levelname)s | %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger()


# parser = argparse.ArgumentParser()
# parser.add_argument('--n_jobs', default=50, type=int, help='Number of jobs to submit')
# args = parser.parse_args()
# print(json.dumps(vars(args), sort_keys=True, indent=4))
# n_jobs = args.n_jobs

N_JOBS = 100
n_jobs = N_JOBS


df = pd.read_csv('overview_file_cohortsEM-ling1.csv')
selected_experiment = 'experiments/massc/avg_kw21/20201126_043826'
# selected_experiment = 'experiments/massc/avg_kw21/20201126_043826/SSC-WSC_test_predictions.pkl'
# with open(selected_experiment, 'rb') as f:
#     predictions = pickle.load(f)
# list_studies = list(predictions.keys())
list_studies = sorted(os.listdir(os.path.join(selected_experiment, 'predictions', 'ssc-wsc_test')))
df_preds = pd.read_csv(os.path.join(selected_experiment, 'SSC-WSC_test_results.csv'), index_col=0)
df_preds = pd.merge(left=df_preds, right=df[['ID', 'Diagnosis', 'Label']], how='left', right_on='ID', left_on='SubjectID')
df_preds['ID_Scrub'] = df_preds['SubjectID'].apply(lambda row: row.split(' ')[0])
df_preds['Diagnosis'] = df_preds.apply(lambda x: x.Diagnosis.replace("'", ''), axis=1)
# select only narcoleptics
df_preds = df_preds.query('Label == 0')
list_studies = [study for study in list_studies if (df_preds.SubjectID == study[6:].split('.')[0]).any()]

file_chunks = [[str(s) for s in arr] for arr in np.array_split(list_studies, n_jobs)]


def run_chunk(chunk):
    # print(chunk)

    for FileID in file_chunks[chunk]:
        # FileID = 'C2157_4 175232.h5'
        # FileID = 'preds_SSC_NARCO_5938_1.pkl'

        # subjectID = FileID.split('.')[0].split(' ')[0]
        fullID = FileID[6:].split('.')[0]
        subjectID = FileID[6:].split('.')[0].split(' ')[0]
        diagnosis = df_preds.query(f'SubjectID == "{fullID}"').Diagnosis.tolist()[0]
        _title = f'{subjectID} | {diagnosis}'
        _save_path = f'/scratch/users/alexno/sleep-staging/figures/spectral-hypnograms/90s/{subjectID}'

        with open(os.path.join(selected_experiment, 'predictions', 'ssc-wsc_test', FileID), 'rb') as pkl:
            predictions = pickle.load(pkl)
        seqs = predictions["seq_nr"]
        interval = 0.5  # minutes
        m_factor = int(5 / interval)
        seqs = np.array([s * m_factor + step for s in seqs for step in range(m_factor)])
        seqs_min = seqs.min()
        seqs_max = seqs.max()

        for seq_nr in tqdm(range(seqs_min + 1, seqs_max), desc=f'{subjectID}'):
            if (seq_nr - 1 in seqs) and (seq_nr in seqs) and (seq_nr + 1 in seqs):
                title = _title + f' | Epoch nr. {seq_nr}/{seqs_max}'
                save_path = _save_path + f'/{subjectID}_seq{seq_nr:04}.png'
                extended_epoch_view(
                    FileID,
                    predictions,
                    seq_nr=seq_nr,
                    interval=0.5,
                    title=title,
                    verbose=True,
                    save_path=save_path
                )

def run_cli():

    logger.info(f"Submitting {n_jobs} job(s) for {len(list_studies)} EDF files...")

    for idx, current_chunk in enumerate(file_chunks):

        # run_chunk(0)
        # return
        log_filename = f'/home/users/alexno/sleep-staging/logs/spectral-hypnograms/spectral-hypnograms-90s_{idx}'

        # if idx < 8:
        content = f"""#!/bin/bash
#
#SBATCH --job-name="{idx:03}"
#SBATCH -p mignot,owners,normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output="{log_filename}.out"
#SBATCH --error="{log_filename}.err"
##################################################

source $PI_HOME/miniconda3/bin/activate
conda activate pt1.7
cd $HOME/sleep-staging

python -c 'from scripts.cli_plot_sequences import run_chunk; run_chunk({idx})'
"""
        with tempfile.NamedTemporaryFile(delete=False) as j:
            j.write(content.encode())
        os.system("sbatch {}".format(j.name))
        # break


    return None


if __name__ == '__main__':
    run_cli()
