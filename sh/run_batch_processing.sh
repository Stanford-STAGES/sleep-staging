# python batch_processing.py --data_dir /oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/WSC_EDF/ --out_dir data/full_length/ssc_wsc/raw/train --encoding_type raw --seq_len 10 --overlap 5 --fs 128
# python batch_processing.py --data_dir /oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/WSC_EDF/ --out_dir data/full_length/ssc_wsc/raw/train --encoding_type raw --seq_len 10 --overlap 5 --fs 128 --slice 0:500
# python batch_processing.py --data_dir /oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/WSC_EDF/ --out_dir data/full_length/ssc_wsc/raw/train --encoding_type raw --seq_len 10 --overlap 5 --fs 128 --slice 500:1000
# python batch_processing.py --data_dir /oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/WSC_EDF/ --out_dir data/full_length/ssc_wsc/raw/train --encoding_type raw --seq_len 10 --overlap 5 --fs 128 --slice 1000:1500
# python batch_processing.py --data_dir /oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/WSC_EDF/ --out_dir data/full_length/ssc_wsc/raw/train --encoding_type raw --seq_len 10 --overlap 5 --fs 128 --slice 1500:2000
# python batch_processing.py --data_dir /oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/WSC_EDF/ --out_dir data/full_length/ssc_wsc/raw/train --encoding_type raw --seq_len 10 --overlap 5 --fs 128 --slice 2000:2500
# python batch_processing.py --data_dir /oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/WSC_EDF/ --out_dir data/full_length/ssc_wsc/raw/train --encoding_type raw --seq_len 10 --overlap 5 --fs 128 --slice 2500:

# python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/Korea (KHC)/SOMNO/" \
#                            --out_dir ./data/khc/raw/5min/ \
#                            --cohort khc \
#                            --encoding_type raw \
#                            --seq_len 10 \
#                            --fs 128 \
#                            --test \
#                            --n_jobs 10

# RUN SSC TRAIN
python -m preprocessing.batch_processing --data_dir "/oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/SSC/NARCO" \
                           --out_dir ./data/all_train \
                           --cohort ssc \
                           --encoding_type raw \
                           --seq_len 10 \
                           --overlap 5 \
                           --fs 128 \
                           --n_jobs 20

# RUN WSC TRAIN
# python -m preprocessing.batch_processing --data_dir "/oak/stanford/groups/mignot/psg/WSC_EDF/" \
#                            --out_dir ./data/all_train \
#                            --cohort wsc \
#                            --encoding_type raw \
#                            --seq_len 10 \
#                            --overlap 5 \
#                            --fs 128 \
#                            --n_jobs 20

# RUN SSC TEST
# python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/SSC/APOE,/oak/stanford/groups/mignot/psg/SSC/NARCO" \
#                            --out_dir ./data/test/raw/ \
#                            --cohort ssc \
#                            --encoding_type raw \
#                            --seq_len 10 \
#                            --fs 128 \
#                            --test \
#                            --n_jobs 20

# RUN WSC TEST
# python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/WSC_EDF/" \
#                            --out_dir ./data/test/raw/ \
#                            --cohort wsc \
#                            --encoding_type raw \
#                            --seq_len 10 \
#                            --fs 128 \
#                            --test \
#                            --n_jobs 20

# RUN KHC
# python -m preprocessing.batch_processing --data_dir data/khc/edf \
#                                          --out_dir data/khc/raw/ \
#                                          --cohort khc \
#                                          --encoding_type raw \
#                                          --seq_len 10 \
#                                          --fs 128 \
#                                          --test \
#                                          --n_jobs 10

# RUN IHC
# python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/Italy (IHC)/Non_NT1,/oak/stanford/groups/mignot/psg/Italy (IHC)/NT1" \
#                            --out_dir ./data/test/raw/ \
#                            --cohort ihc \
#                            --encoding_type raw \
#                            --seq_len 10 \
#                            --fs 128 \
#                            --test \
#                            --n_jobs 10
# python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/Korea (KHC)/SOMNO" --encoding_type cc --seq_len 1200 --overlap 0 --fs 100 --test
# python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/ISRC" --encoding_type raw --seq_len 10 --overlap 0 --fs 128 --test
# python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/SSC/APOE" --encoding_type cc --seq_len 1200 --overlap 0 --fs 100 --test
# python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/WSC_EDF" --encoding_type cc --seq_len 1200 --overlap 0 --fs 100 --test
# python batch_processing.py --data_dir "/oak/stanford/groups/mignot/psg/ISRC" --encoding_type cc --seq_len 1200 --overlap 0 --fs 100 --test

# RUN AHC
# python -m preprocessing.batch_processing --data_dir data/ahc/edf \
#                                          --out_dir data/ahc/raw/ \
#                                          --cohort ahc \
#                                          --encoding_type raw \
#                                          --seq_len 10 \
#                                          --fs 128 \
#                                          --test \
#                                          --n_jobs 10

# # CFS
# python -m preprocessing.batch_processing --data_dir data/cfs/edf \
#                                          --out_dir data/all_train \
#                                          --cohort cfs \
#                                          --encoding_type raw \
#                                          --seq_len 10 \
#                                          --overlap 5 \
#                                          --fs 128 \
#                                          --n_jobs 10

# # CHAT
# python -m preprocessing.batch_processing --data_dir data/chat/edf \
#                                          --out_dir data/all_train \
#                                          --cohort chat \
#                                          --encoding_type raw \
#                                          --seq_len 10 \
#                                          --overlap 5 \
#                                          --fs 128 \
#                                          --n_jobs 20

# # MrOS
# python -m preprocessing.batch_processing --data_dir data/mros/edf \
#                                          --out_dir data/all_train \
#                                          --cohort mros \
#                                          --encoding_type raw \
#                                          --seq_len 10 \
#                                          --overlap 5 \
#                                          --fs 128 \
#                                          --n_jobs 20

# # MESA
# python -m preprocessing.batch_processing --data_dir data/mesa/edf \
#                                          --out_dir data/all_train \
#                                          --cohort mesa \
#                                          --encoding_type raw \
#                                          --seq_len 10 \
#                                          --overlap 5 \
#                                          --fs 128 \
#                                          --n_jobs 20

# # SHHS
# python -m preprocessing.batch_processing --data_dir data/shhs/edf \
#                                          --out_dir data/all_train \
#                                          --cohort shhs \
#                                          --encoding_type raw \
#                                          --seq_len 10 \
#                                          --overlap 5 \
#                                          --fs 128 \
#                                          --n_jobs 20
