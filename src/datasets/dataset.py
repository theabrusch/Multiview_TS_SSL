from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.nn import functional as F
import numpy as np
from src.datasets.simulated_data import cpc_data_simulator, multiview_data_simulator, finetuning_simulator

def get_datasets(data_path, batch_size, pretraining_setup, combine_all = False, subsample=False):
    train = torch.load(data_path + 'train.pt')
    val = torch.load(data_path + 'val.pt')
    test = torch.load(data_path + 'test.pt')

    channels = train['samples'].shape[1]
    time_length = train['samples'].shape[2]
    if pretraining_setup == 'cpc':
        time_length = train['samples'].shape[2] // 2
    num_classes = len(train['labels'].unique())

    if subsample:
        train_idx = np.random.choice(np.arange(len(train['samples'])), size=100, replace=False)
        train = {'samples': train['samples'][train_idx], 'labels': train['labels'][train_idx]}
    
    if combine_all:
        train = {'samples': torch.cat((train['samples'], val['samples'])), 'labels': torch.cat((train['labels'], val['labels']))}
        train_dset = SSL_dataset(train['samples'], train['labels'], pretraining_setup=pretraining_setup)
        train_loader = DataLoader(train_dset, batch_size = batch_size, shuffle = True, drop_last=False)

        val_dset = SSL_dataset(val['samples'], val['labels'], pretraining_setup=pretraining_setup)
        val_loader = DataLoader(val_dset, batch_size = batch_size, drop_last=False)
        return train_loader, val_loader, None, (channels, time_length, num_classes)

    traindset = SSL_dataset(train['samples'], train['labels'], pretraining_setup=pretraining_setup)
    train_loader = DataLoader(traindset, batch_size = batch_size, shuffle = True, drop_last=False)

    val_dset = SSL_dataset(val['samples'], val['labels'], pretraining_setup=pretraining_setup)
    test_dset = SSL_dataset(test['samples'], test['labels'], pretraining_setup=pretraining_setup)
    val_loader = DataLoader(val_dset, batch_size = batch_size, drop_last=False)
    test_loader = DataLoader(test_dset, batch_size = batch_size, drop_last=False)

    return train_loader, val_loader, test_loader, (channels, time_length, num_classes)

def get_simulated_data_pretraining(simulator_type, pretraining_setup, samples, batchsize, n_sources = [5,5], groups_of_dep_var = [8, 2], n_states = 1000, sigma = 0.5, fs = 100, length = 30):
    if simulator_type == 'simulated_cpc':
        simulator = cpc_data_simulator(n_sources, groups_of_dep_var, n_states, sigma, fs, length*2)
    elif simulator_type == 'simulated_multiview':
        if isinstance(n_sources, list):
            n_sources = np.sum(n_sources)
        if isinstance(groups_of_dep_var, list):
            groups_of_dep_var = np.sum(groups_of_dep_var)
        simulator = multiview_data_simulator(n_sources, groups_of_dep_var, n_states, sigma, fs, 2*length)
        
    train = torch.Tensor(simulator.generate(samples[0])).transpose(1,2)
    val = torch.Tensor(simulator.generate(samples[1])).transpose(1,2)

    train_dset = SSL_dataset(train, torch.zeros(samples[0]), pretraining_setup=pretraining_setup)
    val_dset = SSL_dataset(val, torch.zeros(samples[1]), pretraining_setup=pretraining_setup)
    train_loader = DataLoader(train_dset, batch_size = batchsize, shuffle = True, drop_last=False)
    val_loader = DataLoader(val_dset, batch_size = batchsize, drop_last=False)

    channels = train.shape[1]
    time_length = train.shape[2]//2
    num_classes = 1

    return train_loader, val_loader, None, (channels, time_length, num_classes)

def get_simulated_data_finetuning(finetune_setup, samples, batchsize, n_sources = [5,5], groups_of_dep_var = [8, 2], n_states = 2, sigma = 0.5, fs = 100, length = 30):
    if finetune_setup == 'simulated_cpc':
        if len(n_sources) == 1:
            n_sources = [n_sources[0], n_sources[0]]
        if len(groups_of_dep_var) == 1:
            groups_of_dep_var = [groups_of_dep_var[0], groups_of_dep_var[0]]
    elif finetune_setup == 'simulated_multiview':
        if len(n_sources) > 1:
            n_sources = [np.sum(n_sources)]
        if len(groups_of_dep_var) > 1:
            groups_of_dep_var = [np.sum(groups_of_dep_var)]
    simulator = finetuning_simulator(finetune_setup, n_sources, groups_of_dep_var, n_states, sigma, fs, length)

    train = simulator.generate(samples[0])
    val = simulator.generate(samples[1])
    test = simulator.generate(samples[2])

    X_train, y_train = torch.Tensor(train[0]).transpose(1,2), torch.Tensor(train[1]).long()
    X_val, y_val = torch.Tensor(val[0]).transpose(1,2), torch.Tensor(val[1]).long()
    X_test, y_test = torch.Tensor(test[0]).transpose(1,2), torch.Tensor(test[1]).long()

    train_dset = SSL_dataset(X_train, y_train, pretraining_setup='None')
    val_dset = SSL_dataset(X_val, y_val, pretraining_setup='None')
    test_dset = SSL_dataset(X_test, y_test, pretraining_setup='None')

    train_loader = DataLoader(train_dset, batch_size = batchsize, shuffle = True, drop_last=False)
    val_loader = DataLoader(val_dset, batch_size = batchsize, drop_last=False)
    test_loader = DataLoader(test_dset, batch_size = batchsize, drop_last=False)

    channels = X_train.shape[1]
    time_length = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    return train_loader, val_loader, test_loader, (channels, time_length, num_classes)

class SSL_dataset(TensorDataset):
    def __init__(self, X, y, pretraining_setup = None):
        self.X = X
        self.y = y
        self.pretraining_setup = pretraining_setup
        if pretraining_setup == 'multiview':
            self.X = X[:, :, :X.shape[2]//2]

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        
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
    
