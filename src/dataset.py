from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.nn import functional as F
import numpy as np

def get_datasets(data_path, batch_size, pretraining_setup, subsample=False):
    train = torch.load(data_path + 'train.pt')
    val = torch.load(data_path + 'val.pt')
    test = torch.load(data_path + 'test.pt')

    if subsample:
        train_idx = np.random.choice(np.arange(len(train['samples'])), size=100, replace=False)
        train = {'samples': train['samples'][train_idx], 'labels': train['labels'][train_idx]}

    traindset = SSL_dataset(train['samples'], train['labels'], pretraining_setup=pretraining_setup)
    train_loader = DataLoader(traindset, batch_size = batch_size, shuffle = True, drop_last=False)

    val_dset = SSL_dataset(val['samples'], val['labels'], pretraining_setup=pretraining_setup)
    test_dset = SSL_dataset(test['samples'], test['labels'], pretraining_setup=pretraining_setup)
    val_loader = DataLoader(val_dset, batch_size = batch_size, drop_last=False)
    test_loader = DataLoader(test_dset, batch_size = batch_size, drop_last=False)
    
    channels = train['samples'].shape[1]
    time_length = train['samples'].shape[2]
    if pretraining_setup == 'cpc':
        time_length = train['samples'].shape[2] // 2
    num_classes = len(train['labels'].unique())

    return train_loader, val_loader, test_loader, (channels, time_length, num_classes)

class SSL_dataset(TensorDataset):
    def __init__(self, X, y, pretraining_setup = None):
        self.X = X
        self.y = y
        self.pretraining_setup = pretraining_setup

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        if not self.pretraining_setup is None:
            if self.pretraining_setup == 'multiview':
                # partition the dataset into two views
                ch_size = np.random.randint(2, x.size(1)-1)
                random_channels = np.random.rand(x.size(1)).argpartition(x.size(1)-1)
                view_1_idx = random_channels[:ch_size] # randomly select ch_size channels per input
                view_2_idx = random_channels[ch_size:] # take the remaining as the second view
                view_1 = x[:, view_1_idx, :]
                view_2 = x[:, view_2_idx, :]
            elif self.pretraining_setup == 'cpc':
                # partition the x variables into two halves
                time_length = x.size(1)
                half = time_length // 2
                view_1 = x[:, :half]
                view_2 = x[:, half:]
            x = [view_1, view_2]

        return x, y

    def __len__(self):
        return len(self.y)

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
    
