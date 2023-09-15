#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J TS2Vec_HAR
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=64GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 01:00
### added outputs and errors to files
#BSUB -o logs/Output_TS2Vec_HAR_%J.out
#BSUB -e logs/Error_TS2Vec_HAR_%J.err

module load python3/3.9.11

python3 main.py --data_path /work3/theb/timeseries/HAR/ --epochs 40 --finetune_epochs 40 --sample_pretrain_subjs 3 --sample_test_subjs False --batch_size 80 --load_model False --pretrained_model_path outputs/ts2vec_sleepeeg_v_16/pretrained_model.pt --sample_finetune_train_subjs 2 --sample_finetune_val_subjs 1 --save_model True --pretrain True --finetune False --optimize_encoder True --learning_rate 1e-3 --pool 'max' --choose_best True --evaluate_latent_space True
