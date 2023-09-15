#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J bendr
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=64GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 10:00
### added outputs and errors to files
#BSUB -o logs/Output_bendr_%J.out
#BSUB -e logs/Error_bendr_%J.err

module load python3/3.9.11
source timeseries-env/bin/activate

python3 main_bendr.py --data_path 'sleepeeg.yml' --finetune_path 'sleepedf.yml' --sample_pretrain_subjects False
