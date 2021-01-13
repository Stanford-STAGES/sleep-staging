import os
import tempfile
import time


# fmt: off
JOBS = [
    ('ac_rh_ls_lstm_01', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_01 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    ('ac_rh_ls_lstm_02', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_02 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    ('ac_rh_ls_lstm_03', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_03 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    ('ac_rh_ls_lstm_04', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_04 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_05', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_05 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_06', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_06 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_07', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_07 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_08', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_08 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_09', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_09 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_10', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_10 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_11', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_11 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_12', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_12 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_13', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_13 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_14', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_14 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_15', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_15 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ('ac_rh_ls_lstm_16', 'python train.py --model_type stages --model_name ac_rh_ls_lstm_16 --distributed_backend ddp --n_workers 20 --batch_size 16 --gpus 4 --earlystopping_patience 15 --max_epochs 50'),
    # ("mix", "python batch_processing.py --mix --data_dir data/ssc_wsc/cc/5min/single --out_dir data/ssc_wsc/cc/5min/train"),
]
# fmt: on


def submit_job(jobname, experiment):

    content = """#!/bin/bash
#
#SBATCH --job-name={0}
#SBATCH -p mignot,owners,gpu
#SBATCH --time=2-00:00
#SBATCH --cpus-per-task=20
#SBATCH --gpus 4
#SBATCH --output=/home/users/alexno/sleep-staging/logs/{0}.out
#SBATCH --error=/home/users/alexno/sleep-staging/logs/{0}.err
##################################################

source $PI_HOME/miniconda3/bin/activate
conda activate pt1.7
cd $HOME/sleep-staging

{1} --name {0}
"""
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.format(jobname, experiment).encode())
    os.system("sbatch {}".format(j.name))


if __name__ == "__main__":

    for job in JOBS:
        submit_job(job[0], job[1])
        time.sleep(1)

    print("All jobs have been submitted!")
