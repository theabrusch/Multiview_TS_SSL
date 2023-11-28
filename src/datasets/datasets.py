import torch
from torch.utils.data import TensorDataset
from torch.nn import functional as F
import numpy as np
from src.datasets.simulated_data import pretraining_data_simulator, finetuning_simulator
from sklearn.model_selection import train_test_split
import os

def load_numpy_files(data_path, standardize_channels = True, combine_all = False, subsample = False):
    train = torch.load(data_path + 'train.pt')
    val = torch.load(data_path + 'val.pt')
    if 'ptbxl' in data_path:
        if train['samples'].shape[2] < train['samples'].shape[1]:
            train['samples'] = train['samples'].transpose(2,1)
        if val['samples'].shape[2] < val['samples'].shape[1]:
            val['samples']= val['samples'].transpose(2,1)
    
    # check if path exists to test data
    if os.path.exists(data_path + 'test.pt'):
        test = torch.load(data_path + 'test.pt')
    else:
        test = None
        test_dset = None

    channels = train['samples'].shape[1]
    time_length = train['samples'].shape[2]
    
    num_classes = len(train['labels'].unique())

    if subsample:
        train_idx = np.random.choice(np.arange(len(train['samples'])), size=100, replace=False)
        train = {'samples': train['samples'][train_idx], 'labels': train['labels'][train_idx]}
    
    if combine_all:
        train = {'samples': torch.cat((train['samples'], val['samples'])), 'labels': torch.cat((train['labels'], val['labels']))}
    
    train_dset = SSL_dataset(train['samples'], train['labels'], standardize_channels=standardize_channels)
    val_dset = SSL_dataset(val['samples'], val['labels'], standardize_channels=standardize_channels)
    
    if test is not None:
        test_dset = SSL_dataset(test['samples'], test['labels'], standardize_channels=standardize_channels)
    return train_dset, val_dset, test_dset, (channels, time_length, num_classes)

def load_ninaprodb2(data_path, standardize_channels = True):
    path = data_path 
    files = os.listdir(path)
    subjects = np.unique([file.split('_')[0] for file in files])
    train, val = train_test_split(subjects, test_size=0.2, random_state=42)
    train_data, train_labels = [], []
    val_data, val_labels = [], []
    for file in files:
        subject = file.split('_')[0]
        data = np.load(path + file)
        if subject in train:
            train_data.append(data['windows'])
            train_labels.append(data['labels'])
        elif subject in val:
            val_data.append(data['windows'])
            val_labels.append(data['labels'])
    train_data, train_labels = torch.Tensor(np.concatenate(train_data)).transpose(1,2), torch.Tensor(np.concatenate(train_labels)).long()
    val_data, val_labels = torch.Tensor(np.concatenate(val_data)).transpose(1,2), torch.Tensor(np.concatenate(val_labels)).long()
    train_dset = SSL_dataset(train_data, train_labels, standardize_channels=standardize_channels)
    val_dset = SSL_dataset(val_data, val_labels, standardize_channels=standardize_channels)
    channels = train_data.shape[1]
    time_length = train_data.shape[2]
    num_classes = len(torch.unique(train_labels))
    return train_dset, val_dset, (channels, time_length, num_classes)


def get_simulated_data_pretraining(simulator_type, 
                                   pretraining_setup, 
                                   samples, 
                                   standardize_channels = False, 
                                   normalize_emisison = True,
                                   random_emission_matrix = False, 
                                   n_sources = [5,5], 
                                   groups_of_dep_var = [8, 2], 
                                   sigma = 0.5, 
                                   fs = 100, 
                                   length = 30,
                                   seed = 42):
    if simulator_type == 'simulated_cpc':
        groups_of_dep_var = 5*[2]
        n_sources = len(groups_of_dep_var)*[3]
    elif simulator_type == 'simulated_multiview':
        if isinstance(n_sources, list):
            n_sources = [np.sum(n_sources)]
        if isinstance(groups_of_dep_var, list):
            groups_of_dep_var = [np.sum(groups_of_dep_var)]

    simulator = pretraining_data_simulator(n_sources, groups_of_dep_var, sigma, fs, 2*length, normalize_emission=normalize_emisison, simulator_type=simulator_type, seed=seed)
        
    train = torch.Tensor(simulator.generate(samples[0], random_emission_matrix=random_emission_matrix)).transpose(1,2)
    val = torch.Tensor(simulator.generate(samples[1], random_emission_matrix=random_emission_matrix)).transpose(1,2)
    
    if pretraining_setup == 'multiview':
        train = train[:, :, :train.shape[2]//2]
        val = val[:, :, :val.shape[2]//2]

    train_dset = SSL_dataset(train, torch.zeros(samples[0]), standardize_channels=standardize_channels)
    val_dset = SSL_dataset(val, torch.zeros(samples[1]), standardize_channels=standardize_channels)

    channels = train.shape[1]
    time_length = train.shape[2]//2
    num_classes = 1

    return train_dset, val_dset, (channels, time_length, num_classes)

def get_simulated_data_finetuning(finetune_setup, 
                                  samples, 
                                  normalize_emission = True,
                                  standardize_channels = False, 
                                  n_sources = [6,6], 
                                  groups_of_dep_var = [8, 2], 
                                  n_states = 2, 
                                  sigma = 0.5, 
                                  fs = 100, 
                                  length = 30,
                                  seed = 42):
    if finetune_setup == 'simulated_cpc':
        if len(n_sources) == 1:
            n_sources = [n_sources[0], n_sources[0]]
        if len(groups_of_dep_var) == 1:
            groups_of_dep_var = [groups_of_dep_var[0], groups_of_dep_var[0]]
        groups_of_dep_var = 5*[2]
        n_sources = len(groups_of_dep_var)*[3]
    elif finetune_setup == 'simulated_multiview':
        if len(n_sources) > 1:
            n_sources = [np.sum(n_sources)]
        if len(groups_of_dep_var) > 1:
            groups_of_dep_var = [np.sum(groups_of_dep_var)]
    simulator = finetuning_simulator(finetune_setup, n_sources, groups_of_dep_var, n_states, sigma, fs, length, normalize_emission=normalize_emission, seed = seed)

    train = simulator.generate(samples[0])
    val = simulator.generate(samples[1])
    test = simulator.generate(samples[2])

    X_train, y_train = torch.Tensor(train[0]).transpose(1,2), torch.Tensor(train[1]).long()
    X_val, y_val = torch.Tensor(val[0]).transpose(1,2), torch.Tensor(val[1]).long()
    X_test, y_test = torch.Tensor(test[0]).transpose(1,2), torch.Tensor(test[1]).long()

    train_dset = SSL_dataset(X_train, y_train, standardize_channels=standardize_channels)
    val_dset = SSL_dataset(X_val, y_val, standardize_channels=standardize_channels)
    test_dset = SSL_dataset(X_test, y_test, standardize_channels=standardize_channels)

    channels = X_train.shape[1]
    time_length = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    return train_dset, val_dset, test_dset, (channels, time_length, num_classes)

class SSL_dataset(TensorDataset):
    def __init__(self, X, y, standardize_channels = True):
        self.X = X
        self.y = y
        self.standardize_channels = standardize_channels

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        # normalize each epoch channelwise
        if self.standardize_channels:
            stds = x.std(axis=1, keepdims=True)
            stds[stds == 0.] = 1.
            x = (x - x.mean(axis=1, keepdims=True)) / stds
        
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
    