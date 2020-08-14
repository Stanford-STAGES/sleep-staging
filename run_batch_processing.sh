# python batch_processing.py --data_dir /oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/WSC_EDF/ --encoding_type raw --seq_len 10 --overlap 5 --fs 128
# python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/Korea (KHC)/SOMNO" --encoding_type raw --seq_len 10 --overlap 5 --fs 128 --test
python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/Korea (KHC)/SOMNO" --encoding_type cc --seq_len 1200 --overlap 0 --fs 100 --test
