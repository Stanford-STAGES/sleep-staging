# python main.py --model_type massc --max_epochs 2 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --limit_train_batches 100 --optimizer adam --learning_rate 2e-4 --weight_decay 2e-4
# python train.py --model_type stages --model_name ac_rh_ls_lstm_01 --max_epochs 50 --gpus 4 --distributed_backend ddp --n_jobs -1 --n_workers 20 --precision 16
# python predict.py --model_type massc_v2 --resume_from_checkpoint experiments/massc/20200824_060548/epoch=006_best.ckpt --n_workers 20 --gpus 1
python predict.py --model_type massc_v2 --resume_from_checkpoint experiments/massc/20200824_072803/epoch=012_best.ckpt --n_workers 20 --gpus 1
