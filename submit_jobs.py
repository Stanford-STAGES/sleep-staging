import os
import pdb
import sys
import tempfile

JOBS = [
    ('ac_rh_ls_lstm_01', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_01 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_02', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_02 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_03', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_03 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_04', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_04 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_05', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_05 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_06', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_06 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_07', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_07 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_08', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_08 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_09', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_09 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_10', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_10 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_11', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_11 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_12', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_12 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_13', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_13 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_14', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_14 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_15', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_15 --data_dir ./data/batch_encodings --gpus 1'),
    ('ac_rh_ls_lstm_16', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_16 --data_dir ./data/batch_encodings --gpus 1'),
]


def submit_job(jobname, experiment):

    content = '''#!/bin/bash
#
#SBATCH --job-name={0}
#SBATCH -p mignot,owners,gpu
#SBATCH --time=2-00:00
#SBATCH --cpus-per-task=5
#SBATCH --gres gpu:1
#SBATCH --output=/home/users/alexno/sleep-staging/logs/{0}.out
#SBATCH --error=/home/users/alexno/sleep-staging/logs/{0}.err
##################################################

source $PI_HOME/miniconda3/bin/activate
conda activate stages
cd $HOME/sleep-staging

{1}
'''
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.format(jobname, experiment).encode())
    os.system('sbatch {}'.format(j.name))


if __name__ == '__main__':

    for job in JOBS:
        submit_job(job[0], job[1])

    print('All jobs have been submitted!')
