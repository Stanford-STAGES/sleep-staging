import os
import tempfile
from time import sleep

# fmt: off
JOBS = [
    # ('train_massc', 'python train.py --model_type massc --max_epochs 50 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --optimizer adam --learning_rate 2e-4 --weight_decay 2e-4 --scaling robust'),
    # ('train_utime', 'python train.py --model_type utime --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --n_records 1000 --scaling robust'),
    # ('train_utime', 'python train.py --model_type utime --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --scaling robust'),
    # ('train_utime_n500', 'python train.py --model_type utime --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --n_records 500 --scaling robust'),
    # ('train_utime_n1500', 'python train.py --model_type utime --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --n_records 1500 --scaling robust'),
    # ('train_utime_n2000', 'python train.py --model_type utime --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --n_records 2000 --scaling robust'),
    # ('train_massc_stablesleep', 'python train.py --model_type massc --max_epochs 50 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --scaling robust --data_dir data/full_length/ssc_wsc/raw/train --adjustment 30'),
    # ('massc', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 0 --optimizer adam --learning_rate 1e-3 --earlystopping_patience 10'),
    # ('massc_nornn', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 0 --optimizer adam --learning_rate 1e-3 --n_rnn_units 0 --earlystopping_patience 50'),
    # ('massc_15', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 1e-3 --earlystopping_patience 10'),
    # ('massc_nornn_15', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 1e-3 --n_rnn_units 0 --earlystopping_patience 50'),
    # ('massc_30', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --scaling robust --data_dir data/ssc_wsc/radw/5min --adjustment 30 --optimizer adam --learning_rate 1e-3 --earlystopping_patience 10'),
    # ('massc_nornn_30', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 30 --optimizer adam --learning_rate 1e-3 --n_rnn_units 0 --earlystopping_patience 50'),
    # ('massc_rnn', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 0 --optimizer adam --learning_rate 1e-3 --earlystopping_patience 50'),
    # ('massc_rnn_adj30', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 30 --optimizer adam --learning_rate 1e-3 --earlystopping_patience 50'),
    # ('massc_rnn_adj15', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 1e-3 --earlystopping_patience 50'),

    # ('cv1', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 1e-3 --earlystopping_patience 25 --cv -3 --cv_idx 0 --n_rnn_units 0'),
    # ('cv2', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 1e-3 --earlystopping_patience 25 --cv -3 --cv_idx 1 --n_rnn_units 0'),
    # ('cv3', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 1e-3 --earlystopping_patience 25 --cv -3 --cv_idx 2 --n_rnn_units 0'),
    # ('cv1', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 5e-4 --earlystopping_patience 25 --cv 3 --cv_idx 0 --n_rnn_units 0'),
    ('cv2', 'python train.py --model_type massc --max_epochs 100 --gpus 1 --distributed_backend ddp --n_jobs -1 --n_workers 5 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 5e-4 --earlystopping_patience 25 --cv 3 --cv_idx 1 --n_rnn_units 0'),
    ('cv3', 'python train.py --model_type massc --max_epochs 100 --gpus 1 --distributed_backend ddp --n_jobs -1 --n_workers 5 --batch_size 32 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 5e-4 --earlystopping_patience 25 --cv 3 --cv_idx 2 --n_rnn_units 0'),
    # ('cv1', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 16 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 5e-4 --earlystopping_patience 25 --cv -3 --cv_idx 0 --resume_from_checkpoint experiments/massc/cv1/20201029_081358/last.ckpt'),
    # ('cv3', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --batch_size 16 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 15 --optimizer adam --learning_rate 5e-4 --earlystopping_patience 25 --cv -3 --cv_idx 2 --resume_from_checkpoint experiments/massc/cv3/20201029_074212/last.ckpt'),
]
# fmt: on


def submit_job(jobname, experiment):

    content = """#!/bin/bash
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
conda activate pl
cd $HOME/sleep-staging

{1} --name {0}
"""
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.format(jobname, experiment).encode())
    os.system("sbatch {}".format(j.name))


if __name__ == "__main__":

    for job in JOBS:
        submit_job(job[0], job[1])
        sleep(1)

    print("All jobs have been submitted!")
