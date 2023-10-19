from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.nn import functional as F

def get_datasets(data_path, batch_size):
    train = torch.load(data_path + 'train.pt')
    val = torch.load(data_path + 'val.pt')
    test = torch.load(data_path + 'test.pt')

    TFC_dset = TensorDataset(train['samples'], train['labels'])
    train_loader = DataLoader(TFC_dset, batch_size = batch_size, shuffle = True, drop_last=False)

    val_dset = TensorDataset(val['samples'], val['labels'])
    test_dset = TensorDataset(test['samples'], test['labels'])
    val_loader = DataLoader(val_dset, batch_size = batch_size, drop_last=False)
    test_loader = DataLoader(test_dset, batch_size = batch_size, drop_last=False)
    
    channels = train['samples'].shape[1]
    time_length = train['samples'].shape[2]
    num_classes = len(train['labels'].unique())

    return train_loader, val_loader, test_loader, (channels, time_length, num_classes)

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
    
