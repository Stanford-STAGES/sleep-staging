# MASSC model
python train.py --model_type massc --max_epochs 100 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --optimizer adam --learning_rate 2e-4 --weight_decay 2e-4 --scaling robust

# STAGES model
# python train.py --model_type stages --model_name ac_rh_ls_lstm_01 --max_epochs 50 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --precision 16

# U-Time model
# python $HOME/sleep-staging/train.py --model_type utime --max_epochs 100  --gpus 4 --distributed_backend ddp --n_workers 20 --limit_train_batches 0.2

# python train.py --model_type massc_v2 --max_epochs 50 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --optimizer adam --learning_rate 2e-4 --weight_decay 2e-4
# python train.py --model_type avgloss_massc --max_epochs 50 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --optimizer adam --learning_rate 2e-4 --weight_decay 2e-4 --early_stop_callback True
