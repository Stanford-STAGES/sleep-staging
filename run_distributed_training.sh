# python train.py --model_type massc --max_epochs 50 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --optimizer adam --learning_rate 2e-4 --precision 16 --weight_decay 2e-4
python train.py --model_type stages --model_name ac_rh_ls_lstm_01 --max_epochs 50 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --precision 16
