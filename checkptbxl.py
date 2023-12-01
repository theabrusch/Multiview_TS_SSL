import torch
import numpy as np

data_path = '/Users/theb/Desktop/data/ptbxl/processed/'
train = torch.load(data_path + 'train.pt')
val = torch.load(data_path + 'val.pt')
if 'ptbxl' in data_path:
    if train['samples'].shape[2] < train['samples'].shape[1]:
        train['samples'] = train['samples'].transpose(2,1)
    if val['samples'].shape[2] < val['samples'].shape[1]:
        val['samples']= val['samples'].transpose(2,1)

stds = torch.std(train['samples'], axis = 2)

temp = torch.where(stds == 0.)
print(temp)
train['samples'][11278]