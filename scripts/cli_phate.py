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
    gamma = args.gamma
    savepath = args.savepath
    resolution = args.resolution
    topK = args.topK
    repeat_factor = 30 // resolution

    logger.info('PHATE input arguments:')
    logger.info(f'\tNumber of neighbors: {n_knn}')
    logger.info(f'\tGamma: {gamma}')
    logger.info(f'\tSleep scoring resolution: {resolution} s')
    logger.info(f'\ttopK: {topK}')
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

    logger.info(f'Selecting {topK} controls...')
    topK_controls = pd.concat([
        df_preds[
            (df_preds['Label'] == 0)
            & (df_preds['Support - W'] > 20)
            & (df_preds['Support - N1'] > 20)
            & (df_preds['Support - N2'] > 20)
            & (df_preds['Support - N3'] > 20)
            & (df_preds["Support - REM"] > 20)
        ].sort_values(["Accuracy"], ascending=False).head(topK).reset_index(drop=True),
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

    for idx, selected_record in topK_controls.iterrows():

        fileid = selected_record['FileID']
        diagnosis = selected_record['Diagnosis'].replace("'", "")
#         record = predictions[fileid]
        with open(os.path.join(os.path.dirname(selected_experiment), 'predictions', 'ssc-wsc_test', f'preds_{fileid.split(".")[0]}.pkl'), 'rb') as pkl:
            record = pickle.load(pkl)
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
        elif resolution == 30:
            logits = record['predicted'][mask]
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

    logging.info('Creating PHATE object')
    phate_op = phate.PHATE(n_components=2, n_jobs=-1, random_state=1337, knn=n_knn, gamma=gamma)
    X_phate = phate_op.fit_transform(X)

    logging.info('Plotting embedded data')
    ax = scprep.plot.scatter2d(
        X_phate,
        c=l,
        cmap='Spectral',
        ticks=False,
        label_prefix='PHATE',
        dpi=300,
        s=5,
        fontsize=20
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.text(
        0.99,
        0.01,
        f"PHATE: n_knn={n_knn}, gamma={gamma}",
        transform=ax.transAxes,
        horizontalalignment="right",
        color='k',
        fontsize=20,
    )
    return ax

    logging.info('Saving figure')

    try:
        save_path = f'results/phate/cli_phate_knn{n_knn}_gamma{gamma}_resolution{resolution}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    except:
        save_path = f'results/cli_phate_knn{n_knn}_gamma{gamma}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    logging.info('Running 3D PHATE')
    phate_op.set_params(n_components=3)
    X_phate_3d = phate_op.transform()

    logging.info('Plotting embedded 3D data')
    ax = scprep.plot.scatter3d(
        X_phate_3d,
        c=l,
        cmap="Spectral",
        ticks=False,
        label_prefix="PHATE",
    )
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.text(
#         0.99,
#         0.01,
#         f"PHATE: n_knn={n_knn}, gamma={gamma}",
#         transform=ax.transAxes,
#         horizontalalignment="right",
#         color='k',
#         fontsize=20,
#     )

    logging.info('Saving 3D figure')
    try:
        save_path = f'results/phate/cli_phate_knn{n_knn}_gamma{gamma}_resolution{resolution}_3D.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    except:
        save_path = f'results/cli_phate_knn{n_knn}_gamma{gamma}_3D.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    logging.info('Saving 3D PHATE as GIF')

    scprep.plot.rotate_scatter3d(
        X_phate_3d,
        c=l,
        cmap="Spectral",
        ticks=False,
        label_prefix="PHATE",
        dpi=300,
        s=5,
        fontsize=20,
        filename=f'results/phate/cli_phate_knn{n_knn}_gamma{gamma}_resolution{resolution}_3D.gif'
    )
    logger.info('Finished.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_knn', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--savepath', default=None, type=str)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--topK', default=10, type=int)
    args = parser.parse_args()

    run_cli(args)
