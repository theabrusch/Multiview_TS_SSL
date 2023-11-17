from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from src.datasets.eegdataset import construct_eeg_datasets
import torch
from torch.nn import functional as F
import numpy as np
from src.datasets.simulated_data import cpc_data_simulator, multiview_data_simulator, finetuning_simulator

def get_dataloaders_pretraining(args, subsample=False):
    if not 'sleep' in args.data_path:
        if not 'simulated' in args.data_path:
            dset = args.data_path.split('/')[-2]
            train_dset, val_dset, _, (channels, time_length, num_classes) = load_numpy_files(args.data_path, args.batchsize, pretraining_setup=args.pretraining_setup, combine_all = dset == 'chapman', subsample = subsample)
        else:
            dset = args.data_path
            n_samples = [10000, 1000]
            train_dset, val_dset, (channels, time_length, num_classes) = get_simulated_data_pretraining(dset, args.pretraining_setup, n_samples)
    else:
        dset = args.data_path.split('.')[0]
        train_dset, val_dset, (channels, time_length, num_classes) = construct_eeg_datasets(**vars(args))
    
        
    train_loader = DataLoader(train_dset, batch_size = args.batchsize, shuffle = True, drop_last=False)
    val_loader = DataLoader(val_dset, batch_size = args.batchsize, drop_last=False)
    return train_loader, val_loader, dset, (channels, time_length, num_classes)

def get_dataloaders_finetuning(args, balanced_sampling, sample_generator = None, seed = 42):
    if not 'sleep' in args.data_path:
        if not 'simulated' in args.data_path:
            dset = args.data_path.split('/')[-2]
            train_dset, val_dset, test_dset, (channels, time_length, num_classes) = load_numpy_files(args.data_path, args.batchsize, pretraining_setup='None', combine_all = dset == 'chapman')
        else:
            dset = args.data_path
            n_samples = [10, 10, 10]
            balanced_sampling = False
            train_dset, val_dset, test_dset, (channels, time_length, num_classes) = get_simulated_data_finetuning(dset, n_samples)
    else:
        dset = args.data_path.split('.')[0]
        train_dset, val_dset, test_dset, (channels, time_length, num_classes) = construct_eeg_datasets(**vars(args))
    
    if balanced_sampling:
        sample_weights_train, length_train, sample_weights_val, length_val = get_label_balance(train_dset, val_dset, sample_generator = sample_generator, seed = seed)
        train_loader, val_loader = get_train_val_loaders(train_dset, val_dset, args.batchsize, sample_weights_train, length_train, sample_weights_val, length_val)
    else:
        train_loader = [DataLoader(train_dset, batch_size = args.batchsize, shuffle = True, drop_last=False)]
        val_loader = [DataLoader(val_dset, batch_size = args.batchsize, drop_last=False)]

    test_loader = DataLoader(test_dset, batch_size = args.batchsize, drop_last=False)
    return train_loader, val_loader, test_loader, dset, (channels, time_length, num_classes)

def load_numpy_files(data_path, pretraining_setup, combine_all, subsample = False):
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
        val_dset = SSL_dataset(val['samples'], val['labels'], pretraining_setup=pretraining_setup)
    else:
        train_dset = SSL_dataset(train['samples'], train['labels'], pretraining_setup=pretraining_setup)
        val_dset = SSL_dataset(val['samples'], val['labels'], pretraining_setup=pretraining_setup)
    
    test_dset = SSL_dataset(test['samples'], test['labels'], pretraining_setup=pretraining_setup)
    return train_dset, val_dset, test_dset, (channels, time_length, num_classes)



def get_simulated_data_pretraining(simulator_type, pretraining_setup, samples, n_sources = [5,5], groups_of_dep_var = [8, 2], n_states = 1000, sigma = 0.5, fs = 100, length = 30):
    if simulator_type == 'simulated_cpc':
        groups_of_dep_var = 5*[2]
        n_sources = len(groups_of_dep_var)*[3]
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

    channels = train.shape[1]
    time_length = train.shape[2]//2
    num_classes = 1

    return train_dset, val_dset, (channels, time_length, num_classes)

def get_simulated_data_finetuning(finetune_setup, samples, n_sources = [5,5], groups_of_dep_var = [8, 2], n_states = 2, sigma = 0.5, fs = 100, length = 30):
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

    channels = X_train.shape[1]
    time_length = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    return train_dset, val_dset, test_dset, (channels, time_length, num_classes)

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
        # normalize each epoch channelwise
        x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
        
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
    


def get_label_balance(train_dset, val_dset, sample_generator = True, seed = 42):
    if sample_generator:
        # compute sample weights for train and val sets
        sample_weights_train, length_train = fixed_label_balance(train_dset.y, sample_size = sample_generator, seed = seed)
        if isinstance(sample_generator, list):
            val_seed_generator = [int(sg*1) if not sg is None else sg for sg in sample_generator]
        elif isinstance(sample_generator, int):
            val_seed_generator = int(sample_generator*1)

        sample_weights_val, length_val = fixed_label_balance(val_dset.y, sample_size = val_seed_generator, seed=seed)
    else:
        sample_weights_train, length_train = get_label_balance(train_dset)
        sample_weights_val, length_val = get_label_balance(val_dset)
    
    return sample_weights_train, length_train, sample_weights_val, length_val

def get_train_val_loaders(train_dset, val_dset, batchsize, sample_weights_train = None, length_train = None, sample_weights_val = None, length_val = None):
    train_loader = []
    val_loader = []
    if sample_weights_train is None:
        sample_weights_train = np.ones((len(train_dset), 1))
        length_train = [len(train_dset)]
    if sample_weights_val is None:
        sample_weights_val = np.ones((len(val_dset), 1))
        length_val = [len(val_dset)]

    for i in range(sample_weights_train.shape[1]):
        finetune_sampler = WeightedRandomSampler(sample_weights_train[:,i], int(length_train[i]), replacement=False)
        train_loader.append(DataLoader(train_dset, batch_size=batchsize, sampler=finetune_sampler, num_workers=2))

        finetune_val_sampler = WeightedRandomSampler(sample_weights_val[:,i], int(length_val[i]), replacement=False)
        val_loader.append(DataLoader(val_dset, batch_size=batchsize, sampler=finetune_val_sampler, num_workers=2))
    
    return train_loader, val_loader

def fixed_label_balance(labels, sample_size = None, seed = 42):
    """
    Given a dataset, sample a fixed balanced dataset
    Parameters
    ----------
    dataset
    Returns
    -------
    sample_weights, counts
    """
    labs, counts = np.unique(labels, return_counts=True)
    if isinstance(sample_size, int):
        min_count = [sample_size]
    elif isinstance(sample_size, list):
        min_count = sample_size
    else:
        min_count = [np.min(counts)]
    w = 1

    sample_weights = np.zeros((len(labels), len(min_count)))
    for i, lab in enumerate(labs):
        for j, samp in enumerate(min_count):
            if samp is None or samp == 'None':
                samp = np.min(counts)
            # randomly sample min_count examples from each class and
            # assign them a weight of 1/min_count
            idx = np.where(labels == lab)[0]
            samp = samp if len(idx) > samp else len(idx)
            np.random.seed(seed+i+samp)
            idx = np.random.choice(idx, samp, replace=False)
            sample_weights[idx, j] = w

    return sample_weights, sample_weights.sum(axis = 0)
