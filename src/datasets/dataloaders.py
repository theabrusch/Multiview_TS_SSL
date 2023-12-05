from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from src.datasets.eegdataset import construct_eeg_datasets
from src.datasets.datasets import get_simulated_data_finetuning, get_simulated_data_pretraining, load_numpy_files, load_ninaprodb2, load_grapgmyo, load_physionet
import numpy as np

def get_dataloaders_pretraining(args, subsample=False):
    if 'sleep' in args.data_path:
        dset = args.data_path.split('.')[0]
        args.standardize_epochs = 'channelwise'
        train_dset, val_dset, (channels, time_length, num_classes) = construct_eeg_datasets(**vars(args))
    elif 'simulated' in args.data_path:
        dset = args.data_path
        n_samples = [10000, 1000]
        train_dset, val_dset, (channels, time_length, num_classes) = get_simulated_data_pretraining(dset, args.pretraining_setup, n_samples, standardize_channels=args.standardize_channels, random_emission_matrix=args.random_emission_matrix)
    elif 'ninaprodb2' in args.data_path:
        dset = args.data_path.split('/')[-2]
        train_dset, val_dset, (channels, time_length, num_classes) = load_ninaprodb2(args.data_path, standardize_channels=args.standardize_channels)
    elif 'physionet2021' in args.data_path:
        dset = args.data_path.split('/')[-2]
        train_dset, val_dset, (channels, time_length, num_classes) = load_physionet(args.data_path, standardize_channels= args.standardize_channels)
    else:
        dset = args.data_path.split('/')[-2]
        # uniform method for loading ecg datasets
        train_dset, val_dset, _, (channels, time_length, num_classes) = load_numpy_files(args.data_path, combine_all = dset == 'chapman', standardize_channels= args.standardize_channels, subsample = subsample) 
    
    if args.pretraining_setup == 'cpc':
        time_length = time_length // 2

    train_loader = DataLoader(train_dset, batch_size = args.batchsize, shuffle = True, drop_last=False, num_workers=2)
    val_loader = DataLoader(val_dset, batch_size = args.batchsize, drop_last=False, num_workers=2)
    return train_loader, val_loader, dset, (channels, time_length, num_classes)

def get_dataloaders_finetuning(args, balanced_sampling, sample_generator = None, seed = 42):
    if 'sleep' in args.data_path:
        dset = args.data_path.split('.')[0]
        args.standardize_epochs = 'channelwise'
        train_dset, val_dset, test_dset, (channels, time_length, num_classes) = construct_eeg_datasets(**vars(args))
    elif 'simulated' in args.data_path:
        dset = args.data_path
        n_samples = [10000, 1000, 1000]
        balanced_sampling = False
        train_dset, val_dset, test_dset, (channels, time_length, num_classes) = get_simulated_data_finetuning(dset, n_samples, standardize_channels=args.standardize_channels, seed = args.seed)
    elif 'grabgmyo' in args.data_path:
        dset = args.data_path.split('/')[-2]
        train_dset, val_dset, test_dset, (channels, time_length, num_classes) = load_grapgmyo(args.data_path,  window_size = args.window_size, overlap = args.overlap, standardize_channels=args.standardize_channels)
    else:
        dset = args.data_path.split('/')[-2]
        train_dset, val_dset, test_dset, (channels, time_length, num_classes) = load_numpy_files(args.data_path, standardize_channels= args.standardize_channels)
        
    if balanced_sampling:
        sample_weights_train, length_train, sample_weights_val, length_val = get_label_balance(train_dset, val_dset, sample_generator = sample_generator, seed = seed)
        train_loader, val_loader = get_train_val_loaders(train_dset, val_dset, args.batchsize, sample_weights_train, length_train, sample_weights_val, length_val)
    else:
        train_loader = [DataLoader(train_dset, batch_size = args.batchsize, shuffle = True, drop_last=False, num_workers=2)]
        val_loader = [DataLoader(val_dset, batch_size = args.batchsize, drop_last=False, num_workers=2)]

    test_loader = DataLoader(test_dset, batch_size = args.batchsize, drop_last=False, num_workers=2)
    return train_loader, val_loader, test_loader, dset, (channels, time_length, num_classes)



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
    if isinstance(sample_size, int) or isinstance(sample_size, float):
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
