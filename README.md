# Multi-view self-supervised learning for multivariate variable-channel time series
Implementation of the code for "Multi-view self-supervised learning for multivariate variable-channel time series" (https://arxiv.org/abs/2307.09614)

## Downloading and preprocessing the data 
Initially, download the The PhysioNet/Computing in Cardiology Challenge 2018 data (https://physionet.org/content/challenge-2018/1.0.0/) for pretraining and the Sleep Cassette data (https://www.physionet.org/content/sleep-edfx/1.0.0/). 
Run the following lines with correct root paths and desired output folders to preprocess the data to follow the correct format:
```
python3 preprocess_ps18.py --root_folder /path/to/challenge-2018/1.0.0/training/ --out_folder /path/to/ps18/
```
```
python3 preprocess_sleepedf.py --root_folder /path/to/sleep-casette/ --out_folder /path/to/sleepedf/
```
Following this, update the field toplevel in sleepedf.yml and sleepps18.yml to the chosen output paths.

## Pretraining the models
To pretrain the the models run the following code, changing pretraining_setup to either MPNN or nonMPNN. The loss can be chosen as "time_loss", "contrastive" or "COCOA". See the paper for further details. 
```
python3 main.py --pretrain_epochs 10 --batchsize 64 --pretrain True --finetune False --learning_rate 1e-3 --sample_pretrain_subjects False --loss 'time_loss' --pretraining_setup 'MPNN'
```

Feel free to contact me, Thea, at theb@dtu.dk upon any questions. 
I plan to update the repo with cleaner versions of the code during the fall of 2023 (i.e. I will not change major functionality, but simply remove redundant code and improve the commenting). 
