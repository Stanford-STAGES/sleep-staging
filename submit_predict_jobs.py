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
    # ('massc_30', 'python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --scaling robust --data_dir data/ssc_wsc/raw/5min --adjustment 30 --optimizer adam --learning_rate 1e-3 --earlystopping_patience 10'),
    # ('15s', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/20201024_054652/epoch=043-eval_loss=0.58.ckpt'),
    # ('cv3.1', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv3_cvidx0/20201116_070633'),
    # ('cv3.2', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv3_cvidx1/20201116_070633'),
    # ('cv3.3', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv3_cvidx2/20201116_070633'),
    # ('cv2.1', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv2_cvidx0/20201116_070649'),
    # ('cv2.2', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv2_cvidx1/20201116_070633'),
    # ('pcv-3.1', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/cv-3_cvidx0/20201116_173909'),
    # ('cv-3.2', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv-3_cvidx1/20201116_152130'),
    # ('cv-3.3', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv-3_cvidx2/20201116_152129'),
    # ('cv-4.1', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv-4_cvidx0/20201116_152129'),
    # ('pcv-4.2', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/cv-4_cvidx1/20201116_181635'),
    # ('cv-4.3', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv-4_cvidx2/20201116_152133'),
    # ('cv-4.3', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv-4_cvidx3/20201116_155556'),
    # ('cv1.1', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/cv1_cvidx0/20201113_134910'),
    # ('pkw3', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/kw3/20201114_001956'),
    # ('pkw5', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/kw5/20201112_135059'),
    # ('pkw7', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/kw7/20201112_134951'),
    # ('pkw9', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/kw9/20201112_134951'),
    # ('pkw11', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/kw11/20201112_145223'),
    # ('pkw13', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/kw13/20201113_130623'),
    # ('pkw15', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/kw15/20201112_145045'),
    # ('pkw17', 'python predict.py --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --resume_from_checkpoint experiments/massc/kw17/20201113_125807'),
    # ('pkw19', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw19/20201120_072840'),
    # ('pkw21', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw21/20201119_072314'),
    # ('pkw23', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw23/20201121_060956'),
    # ('pkw25', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw25/20201121_061018'),
    # ('pkw27', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw27/20201121_072454'),
    # ('pkw29', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw29/20201121_072425'),
    ('pkw31', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw31/20201124_065129'),
    # ('pkw33', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw33/20201123_030125'),
    # ('pkw35', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw35/20201121_061018'),
    # ('pkw37', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw37/20201123_030125'),
    # ('pkw39', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw39/20201123_030130'),
    # ('pkw41', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/kw41/20201123_092604'),
    # ('patt15', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/att15/20201124_080608'),
    # ('patt30', 'python predict.py --gpus 2 --distributed_backend ddp --n_jobs -1 --n_workers 10 --resume_from_checkpoint experiments/massc/att30/20201124_080609'),
]
# fmt: on


def submit_job(jobname, experiment):

    content = """#!/bin/bash
#
#SBATCH --job-name={0}
#SBATCH -p mignot,owners,gpu
#SBATCH --time=2-00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres gpu:2
#SBATCH --output=/home/users/alexno/sleep-staging/logs/{0}.out
#SBATCH --error=/home/users/alexno/sleep-staging/logs/{0}.err
##################################################

source $PI_HOME/miniconda3/bin/activate
conda activate pt1.7
cd $HOME/sleep-staging

{1}
"""
    with tempfile.NamedTemporaryFile(delete=False) as j:
        j.write(content.format(jobname, experiment).encode())
    os.system("sbatch {}".format(j.name))


if __name__ == "__main__":

    for job in JOBS:
        submit_job(job[0], job[1])
        sleep(1)

    print("All jobs have been submitted!")
