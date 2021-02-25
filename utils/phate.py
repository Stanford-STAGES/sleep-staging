import argparse
import itertools
import logging
import os
import pickle
import pprint
from glob import glob
os.chdir(os.path.expandvars('$HOME/sleep-staging'))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import scprep
import seaborn as sns
import umap
import umap.plot
from h5py import File
from tqdm.notebook import tqdm

sns.set(context='paper', rc={'figure.figsize':(11,10)})

HYP_DICT = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger()

df = pd.read_csv('overview_file_cohortsEM-ling1.csv')


def run_cli(args):
    
    n_knn = args.n_knn
    savepath = args.savepath
    resolution = args.resolution
    repeat_factor = 30 // resolution
    
    logger.info('PHATE input arguments:')
    logger.info(f'\tNumber of neighbors: {n_knn}')
    logger.info(f'\tSleep scoring resolution: {resolution} s')
    logger.info(f'\tRepeat factor: {repeat_factor}')
    logger.info(f'\tSaving figures at: {savepath}\n')
    
    selected_experiment = 'experiments/massc/avg_kw21/20201126_043826/SSC-WSC_test_predictions.pkl'
    with open(selected_experiment, 'rb') as f:
        predictions = pickle.load(f)
    list_studies = list(predictions.keys())
    df_preds = pd.read_csv(os.path.join(os.path.dirname(selected_experiment), 'SSC-WSC_test_results.csv'), index_col=0)
    df_preds = pd.merge(left=df_preds, right=df[['ID', 'Diagnosis', 'Label']], how='left', right_on='ID', left_on='SubjectID')
    logger.info(f"Unique classes: {df_preds['Diagnosis'].unique()}")
    logger.info(f"Window unique classes: {df_preds['Window'].unique()}")
    logger.info(f"Case unique classes: {df_preds['Case'].unique()}")
    
    logger.info('Selecting top10 controls...')
    top10_controls = pd.concat([
        df_preds[
            (df_preds['Label'] == 0)
            & (df_preds['Support - W'] > 20) 
            & (df_preds['Support - N1'] > 20) 
            & (df_preds['Support - N2'] > 20) 
            & (df_preds['Support - N3'] > 20) 
            & (df_preds["Support - REM"] > 20)
        ].sort_values(["Accuracy"], ascending=False).head(10).reset_index(drop=True),
    ])
    
    X = []
    t = []
    p = []
    l = []
    d = []
    fid = []
    idx = []
    idx_5min = []
    fid_vec = []
    fid_counter = 0
    
    for idx, selected_record in top10_controls.iterrows():
        
        fileid = selected_record['FileID']
        diagnosis = selected_record['Diagnosis'].replace("'", "")
        record = predictions[fileid]
        mask = record['true'].sum(axis=1) == 1
        target = (record['true'][mask].argmax(axis=1)).repeat(repeat_factor)
        labels = np.array([HYP_DICT[t] for t in target])
        labels_disease = [diagnosis for _ in range(len(target))]
        seq_nr = record['seq_nr'].repeat(10)[mask].repeat(repeat_factor) # each 30 s is part of a 5 min sequence
        if resolution == 1:
            logits = record['logits'][mask.repeat(30)]
        elif resolution in [3, 5, 10, 15]:
            try:
                logits = record[f'yhat_{resolution}s'][mask.repeat(repeat_factor)]
            except:
                logger.info(f'Available resolutions: {record.keys()}')
                return
        predicted_class = np.array([HYP_DICT[t] for t in logits.argmax(axis=1)])

        X.append(logits)
        t.append(target)
        l.append(labels)
        d.extend(labels_disease)
        fid.extend([fileid for _ in range(len(target))])
        fid_vec.extend([fid_counter for _ in range(len(target))])
        p.extend(predicted_class)
        fid_counter += 1

    X = np.concatenate(X, axis=0)
    y = np.concatenate(t, axis=0)
    l = np.concatenate(l, axis=0)
    data = pd.DataFrame({
        'Stage': l,
        'FileID': fid,
        'Pred_stage': p,
    })
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_knn', type=int)
    parser.add_argument('--savepath', type=str)
    parser.add_argument('--resolution', type=int)
    args = parser.parse_args()
    
    run_cli(args)