#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J multiview_finetune_flatten_false
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=256GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 24:00
### added outputs and errors to files
#BSUB -o logs/Output_multiview_finetune_flatten_false_%J.out
#BSUB -e logs/Error_multiview_finetune_flatten_false_%J.err

module load python3/3.9.11

python3 main_multiview.py --data_path sleepeeg.yml --finetune_path 'same' --pretrain_epochs 20 --finetune_epochs 40 --batchsize 32 --load_model True --save_model True --pretrain False --finetune True --optimize_encoder False --learning_rate 1e-3 --pool 'max' --choose_best True --evaluate_latent_space False --multi_channel_setup 'sample_channel' --sample_pretrain_subjects False --sample_finetune_train_subjects 16 --sample_finetune_val_subjects 4 --target_batchsize 32 --encoder 'wave2vec' --ft_learning_rate 1e-4 --sample_test_subjects False --weight_decay 1e-2 --balanced_sampling 'finetune' --feat_do 0.4 --hidden_channels 256 --output_path 'pretrained_outputs' --pretrained_model_path 'pretrained_models/pretrained_outputs/MultiView_sleepeeg_small_pretrained_timeloss/pretrained_model.pt'
