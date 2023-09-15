from torch.utils.data import Dataset, DataLoader
import torch
from src.augmentations import frequency_augmentation, time_augmentation
from torch.nn import functional as F
import numpy as np

def get_datasets(data_path, batchsize, ssl_mode = 'TS2Vec', downsample = False, abs_budget = False, finetune_mode = False, sample_channel = False, **kwargs):
    train = torch.load(data_path + 'train.pt')
    val = torch.load(data_path + 'val.pt')
    test = torch.load(data_path + 'test.pt')
    if downsample:
        train_idx = np.random.choice(np.arange(len(train['samples'])), size = 128, replace=False)
        train['samples'] = train['samples'][train_idx]
        train['labels'] = train['labels'][train_idx]



    dset = data_path.split('/')[-2]
    if ssl_mode == 'TFC':
        pretrain_dset = TFC_Dataset(train['samples'], train['labels'], dset, abs_budget=abs_budget, fine_tune_mode=finetune_mode, sample_channel=sample_channel)
        val_dset = TFC_Dataset(val['samples'], val['labels'], dset = dset, abs_budget = abs_budget, fine_tune_mode=finetune_mode, sample_channel=sample_channel)
        test_dset = TFC_Dataset(test['samples'], test['labels'], dset = dset, test_mode = True, fine_tune_mode=False, sample_channel=sample_channel)
    elif ssl_mode == 'TS2Vec':
        pretrain_dset = TS2Vec_Dataset(train['samples'], train['labels'], sample_channel=sample_channel)
        val_dset = TS2Vec_Dataset(val['samples'], val['labels'], sample_channel=sample_channel)
        test_dset = TS2Vec_Dataset(test['samples'], test['labels'], sample_channel=sample_channel)

    train_loader = DataLoader(pretrain_dset, batch_size = batchsize, shuffle = True, drop_last=False)
    val_loader = DataLoader(val_dset, batch_size = batchsize, drop_last=False)
    test_loader = DataLoader(test_dset, batch_size = batchsize, drop_last=False)

    return train_loader, val_loader, test_loader, (pretrain_dset.channels, pretrain_dset.time_length, pretrain_dset.num_classes)

def get_dset_info(data_path = None, X = None, dset = None, sample_channel=False):
    if X is None:
        train = torch.load(data_path + 'train.pt')
        X = train['samples']
        dset = data_path.split('/')[-2]
    
    if dset == 'HAR':
        time_length = 206
        if not sample_channel:
            channels = 3
        else:
            channels = X.shape[1]
    else:
        time_length = X.shape[2]
        channels = X.shape[1]
    return channels, time_length   
    


class TFC_Dataset(Dataset):
    def __init__(self, X, Y, dset, sample_channel = False, abs_budget = False, fine_tune_mode = False, test_mode = False):
        super().__init__()

        channels, time_length = get_dset_info(X = X, dset = dset, sample_channel = sample_channel)

        if time_length > X.shape[2]:
             X = F.pad(X[:,:channels,:], (0,int(time_length-X.shape[2])))
        else:
            X = X[:,:channels,:time_length]

        self.X_t = X
        self.Y = Y
        self.time_length = X.shape[2]
        self.num_classes = len(torch.unique(Y))
        
        if not sample_channel:
            self.channels = channels
        else:
            self.num_channels = channels
            # num of channels for the NN to take as input
            self.channels = 1
        self.sample_channel = sample_channel
        if int(torch.max(self.Y)) == self.num_classes:
            self.Y = self.Y-1
        self.test_mode = test_mode
        self.fine_tune_mode = fine_tune_mode

        self.X_f = torch.fft.fft(self.X_t, axis = -1).abs()
        if not test_mode and not fine_tune_mode:
            config = {
                'jitter_scale_ratio': 1.1,
                'jitter_ratio': 0.8,
                'max_seg': 8
            }
            self.X_f_aug = frequency_augmentation(self.X_f, config)
            self.X_t_aug = time_augmentation(self.X_t, config)
    
    def __getitem__(self, idx):
        if not self.test_mode and not self.fine_tune_mode:
            return self.X_t[idx], self.X_f[idx], self.X_t_aug[idx], self.X_f_aug[idx], self.Y[idx]
        elif self.fine_tune_mode:
            return self.X_t[idx], self.X_f[idx], self.X_t[idx], self.X_f[idx], self.Y[idx]
        else:
            return self.X_t[idx], self.X_f[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X_t)
    

class TS2Vec_Dataset(Dataset):
    def __init__(self, X, Y, sample_channel = False):
        super().__init__()

        self.X = X
        self.Y = Y
        self.time_length = X.shape[2]
        self.num_classes = len(torch.unique(Y))
        channels, time_length = self.X.shape[1], self.X.shape[-1]
        self.time_length = time_length
        if not sample_channel:
            self.channels = channels
        else:
            self.num_channels = channels
            # num of channels for the NN to take as input
            self.channels = 1
        self.sample_channel = sample_channel
        if int(torch.max(self.Y)) == self.num_classes:
            self.Y = self.Y-1
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)