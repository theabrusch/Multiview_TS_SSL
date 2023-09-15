#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J Multchannel_sample_channel_HAR
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=8GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 00:30
### added outputs and errors to files
#BSUB -o logs/Output_Multchannel_sample_channel_HAR_%J.out
#BSUB -e logs/Error_Multchannel_sample_channel_HAR_%J.err

module load python3/3.9.11

python3 main.py --data_path /work3/theb/timeseries/HAR/ --epochs 40 --finetune_epochs 40 --batch_size 80 --load_model False --save_model True --pretrain True --finetune True --optimize_encoder False --learning_rate 1e-3 --pool 'max' --choose_best True --evaluate_latent_space True --multi_channel_setup 'sample_channel'
