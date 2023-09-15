import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import os
from fnmatch import fnmatch
import glob
import json
import mne 
from dn3.configuratron import ExperimentConfig
from dn3.data.dataset import EpochTorchRecording, Thinker, Dataset, DatasetInfo
from dn3.transforms.instance import To1020, MappingDeep1010, TemporalInterpolation
import numpy as np
from scipy.signal import filtfilt, butter, sosfiltfilt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def construct_eeg_datasets(data_path, 
                           finetune_path,
                           batchsize, 
                           target_batchsize,
                           standardize_epochs = False,
                           balanced_sampling = 'None',
                           sample_pretrain_subjects = False, 
                           sample_finetune_train_subjects = False,
                           sample_finetune_val_subjects = False,
                           sample_test_subjects = False,
                           exclude_subjects = None,
                           train_mode = 'both',
                           sample_generator = False,
                           bendr_setup = False,
                           load_original_bendr = False,
                           seqclr_setup = False,
                           upsample_bendr = False,
                           seed = None,
                           chunk_duration = '30',
                           **kwargs):
    experiment = ExperimentConfig(data_path)
    dset = data_path.split('/')[-1].strip('.yml').split('_')[0]
    config = experiment.datasets[dset]
    config.normalize = False
    if balanced_sampling == 'pretrain' or balanced_sampling == 'both':
        config.balanced_sampling = True
    else:
        config.balanced_sampling = False

    if bendr_setup and upsample_bendr:
        config.chunk_duration = chunk_duration
        config.upsample = False
    else:
        config.upsample = False
    
    if load_original_bendr:
        config.upsample = True
        config.deep1010 = True
        bendr_setup = False
        standardize_epochs = False
    
    if seqclr_setup:
        config.tlen = int(chunk_duration) + 2
       
    if not exclude_subjects is None:
        config.exclude_people = exclude_subjects
    
    if finetune_path == 'same':
        split_path = data_path.removesuffix('.yml') + '_splits.txt'
        with open(split_path, 'r') as split_file:
            splits = json.load(split_file)
        pretrain_subjects = splits['pretrain']
    else:
        pretrain_subjects = None
    info = DatasetInfo(config.name, config.data_max, config.data_min, config._excluded_people,
                            targets=config._targets if config._targets is not None else len(config._unique_events))
    # construct pretraining datasets
    if train_mode == 'pretrain' or train_mode == 'both':
        print('Loading pre-training data')
        pretrain_thinkers = load_thinkers(config, sample_subjects=sample_pretrain_subjects, subjects = pretrain_subjects)
        pretrain_train_thinkers, pretrain_val_thinkers = divide_thinkers(pretrain_thinkers)
        pretrain_dset, pretrain_val_dset = Dataset(pretrain_train_thinkers, dataset_info=info), Dataset(pretrain_val_thinkers, dataset_info=info)

        aug_config = { 
            'jitter_scale_ratio': 1.1,
            'jitter_ratio': 0.8,
            'max_seg': 8
        }

        if not seqclr_setup:
            pretrain_dset, pretrain_val_dset = EEG_dataset(pretrain_dset, aug_config, standardize_epochs=standardize_epochs), EEG_dataset(pretrain_val_dset, aug_config, standardize_epochs=standardize_epochs)
        else:
            pretrain_dset, pretrain_val_dset = SeqCLR_dataset(pretrain_dset, window_length=int(chunk_duration), standardize_epochs=standardize_epochs), SeqCLR_dataset(pretrain_val_dset, window_length=int(chunk_duration), standardize_epochs=standardize_epochs)

        if config.balanced_sampling:
            if sample_generator:
                sample_weights, counts = fixed_label_balance(pretrain_dset, sample_size = sample_generator, seed=seed)
            else:
                sample_weights, counts = get_label_balance(pretrain_dset)

            pretrain_sampler = WeightedRandomSampler(sample_weights, len(counts) * int(counts.min()), replacement=False)
            pretrain_loader = DataLoader(pretrain_dset, batch_size=batchsize, sampler=pretrain_sampler, num_workers=2)
        else:
            pretrain_loader = DataLoader(pretrain_dset, batch_size=batchsize, shuffle = True, num_workers=2)

        pretrain_loader, pretrain_val_loader = DataLoader(pretrain_dset, batch_size=batchsize, shuffle = True, num_workers=2), DataLoader(pretrain_val_dset, batch_size=batchsize, shuffle = True, num_workers=2)
    else:
        pretrain_loader, pretrain_val_loader = None, None
    
    # construct finetuning dataset
    if train_mode == 'finetune' or train_mode == 'both':
        
        #sample_subjects = int(sample_subjects/2) if sample_subjects else sample_subjects
        if balanced_sampling == 'finetune' or balanced_sampling == 'both':
            config.balanced_sampling = True
        else:
            config.balanced_sampling = False
        if not finetune_path == 'same':
            experiment = ExperimentConfig(finetune_path)
            dset = finetune_path.split('/')[-1].strip('.yml').split('_')[0]
            config = experiment.datasets[dset]
            config.normalize = False

            if balanced_sampling == 'finetune' or balanced_sampling == 'both':
                config.balanced_sampling = True
            else:
                config.balanced_sampling = False

            finetunesubjects, test_subjects = divide_subjects(config, sample_finetune_train_subjects, sample_test_subjects, subjects = None, test_size=config.test_size)
        else:
            config.chunk_duration = str(config.tlen)
            finetunesubjects = splits['finetune']
            test_subjects = splits['test']
        info = DatasetInfo(config.name, config.data_max, config.data_min, config._excluded_people,
                            targets=config._targets if config._targets is not None else len(config._unique_events))

        if bendr_setup and upsample_bendr:
            config.chunk_duration = chunk_duration
        else:
            config.upsample = False

        if load_original_bendr:
            config.upsample = True
            config.deep1010 = True
            bendr_setup = False
            standardize_epochs = False
    

        if seqclr_setup:
            config.tlen = int(chunk_duration)

        print('Loading finetuning data')
        train_subjs, val_subjs = divide_subjects(config, sample_finetune_train_subjects, sample_finetune_val_subjects, subjects = finetunesubjects, test_size=config.val_size)
        finetune_train_thinkers = load_thinkers(config, sample_subjects=False, subjects = train_subjs)
        finetune_val_thinkers = load_thinkers(config, sample_subjects=False, subjects = val_subjs)
        finetune_train_dset, finetune_val_dset = Dataset(finetune_train_thinkers, dataset_info=info), Dataset(finetune_val_thinkers, dataset_info=info)

        if load_original_bendr:
            finetune_train_dset.add_transform(To1020())
            finetune_val_dset.add_transform(To1020())

        aug_config = { 
            'jitter_scale_ratio': 1.1,
            'jitter_ratio': 0.8,
            'max_seg': 8
        }
        if not seqclr_setup:
            finetune_train_dset, finetune_val_dset = EEG_dataset(finetune_train_dset, aug_config, standardize_epochs=standardize_epochs, bendr_setup = bendr_setup), EEG_dataset(finetune_val_dset, aug_config, standardize_epochs=standardize_epochs, bendr_setup=bendr_setup)
        else:
            finetune_train_dset, finetune_val_dset = SeqCLR_dataset(finetune_train_dset, fine_tune_mode=True, window_length=int(config.chunk_duration), standardize_epochs=standardize_epochs), SeqCLR_dataset(finetune_val_dset, fine_tune_mode=True, window_length=int(config.chunk_duration), standardize_epochs=standardize_epochs)
        
        if config.balanced_sampling:
            if sample_generator:
                # compute sample weights for train and val sets
                sample_weights_train, length_train = fixed_label_balance(finetune_train_dset, sample_size = sample_generator, seed = seed)
                if isinstance(sample_generator, list):
                    val_seed_generator = [int(sg*1) if not sg is None else sg for sg in sample_generator]
                elif isinstance(sample_generator, int):
                    val_seed_generator = int(sample_generator*1)

                sample_weights_val, length_val = fixed_label_balance(finetune_val_dset, sample_size = val_seed_generator, seed=seed)
            else:
                sample_weights_train, length_train = get_label_balance(finetune_train_dset)
                sample_weights_val, length_val = get_label_balance(finetune_val_dset)

            finetune_loader = []
            finetune_val_loader = []
            for i in range(sample_weights_train.shape[1]):
                finetune_sampler = WeightedRandomSampler(sample_weights_train[:,i], int(length_train[i]), replacement=False)
                finetune_loader.append(DataLoader(finetune_train_dset, batch_size=target_batchsize, sampler=finetune_sampler, num_workers=2))

                finetune_val_sampler = WeightedRandomSampler(sample_weights_val[:,i], int(length_val[i]), replacement=False)
                finetune_val_loader.append(DataLoader(finetune_val_dset, batch_size=target_batchsize, sampler=finetune_val_sampler, num_workers=2))
        else:
            finetune_loader = DataLoader(finetune_train_dset, batch_size=target_batchsize, shuffle = True, num_workers=2)
            finetune_val_loader = DataLoader(finetune_val_dset, batch_size=target_batchsize, shuffle = True, num_workers=2)
        
        # get test set
        print('Loading test data')
        config.balanced_sampling = False
        test_thinkers = load_thinkers(config, sample_subjects = sample_test_subjects, subjects = test_subjects)
        test_dset = Dataset(test_thinkers, dataset_info=info)

        if load_original_bendr:
            test_dset.add_transform(To1020())

        if not seqclr_setup:
            test_dset = EEG_dataset(test_dset, aug_config, fine_tune_mode=False, standardize_epochs=standardize_epochs, bendr_setup = bendr_setup)
        else:
            test_dset = SeqCLR_dataset(test_dset, fine_tune_mode=True, standardize_epochs=standardize_epochs, window_length=int(config.chunk_duration))
        
        test = False
        if test:
            sample_weights_val, length_val = fixed_label_balance(test_dset, sample_size = 10, seed=seed)
            finetune_sampler = WeightedRandomSampler(sample_weights_train[:,i], int(length_train[i]), replacement=False)
            test_loader = DataLoader(test_dset, batch_size=target_batchsize, shuffle = True, sampler=finetune_sampler, num_workers=2)
        else:
            test_loader = DataLoader(test_dset, batch_size=target_batchsize, shuffle = True, num_workers=2)
        num_classes = len(np.unique(test_dset.dn3_dset.get_targets()))
    else:
        finetune_loader, finetune_val_loader, test_loader, num_classes = None, None, None, 5

    return pretrain_loader, pretrain_val_loader,finetune_loader, finetune_val_loader, test_loader, (len(config.picks), config.tlen*100, num_classes)

def divide_thinkers(thinkers):
    train, val = train_test_split(list(thinkers.keys()), test_size = 0.2, random_state=0)
    train_thinkers = dict()
    for subj in train:
        train_thinkers[subj] = thinkers[subj]
    val_thinkers = dict()
    for subj in val:
        val_thinkers[subj] = thinkers[subj]

    return train_thinkers, val_thinkers

def divide_subjects(config, sample_train, sample_val, test_size = 0.2, subjects = None):
    if subjects is None:
        subjects = os.listdir(config.toplevel)
    train, val = train_test_split(subjects, test_size = test_size, random_state=0)
    if sample_train:
        if sample_train < len(train):
            np.random.seed(0)
            train = np.random.choice(train, sample_train, replace=False)
    if sample_val:
        if sample_val < len(val):
            np.random.seed(0)
            val = np.random.choice(val, sample_val, replace=False)
    return train, val

def fixed_label_balance(dataset, sample_size = None, seed = 42):
    """
    Given a dataset, sample a fixed balanced dataset
    Parameters
    ----------
    dataset
    Returns
    -------
    sample_weights, counts
    """
    labels = dataset.dn3_dset.get_targets()
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

def get_label_balance(dataset):
    """
    Given a dataset, return the proportion of each target class and the counts of each class type
    Parameters
    ----------
    dataset
    Returns
    -------
    sample_weights, counts
    """
    labels = dataset.dn3_dset.get_targets()
    counts = np.bincount(labels)
    train_weights = 1. / torch.tensor(counts, dtype=torch.float)
    sample_weights = train_weights[labels]
    class_freq = counts/counts.sum()
    return sample_weights, len(counts) * int(counts.min())

def load_thinkers(config, sample_subjects = False, subjects = None):
    if subjects is None:
        subjects = os.listdir(config.toplevel)        
    subjects = [subject for subject in subjects if not subject in config.exclude_people]
    if sample_subjects:
        np.random.seed(0)
        subjects = np.random.choice(subjects, sample_subjects, replace = False)
    thinkers = dict()
    for i, subject in enumerate(subjects):
        print('Loading subject', i+1, 'of', len(subjects))
        subj_path = os.path.join(config.toplevel, subject)
        files = glob.glob(f'{subj_path}/*.fif')
        sessions = dict()
        for file in files:
            sess = file.split('/')[-1].strip('.fif')
            recording = construct_epoch_dset(file, config)
            sessions[sess] = recording
        if len(sessions.keys()) > 0:
            thinkers[subject] = Thinker(sessions)
    return thinkers

class _DumbNamespace:
    def __init__(self, d: dict):
        self._d = d.copy()
        for k in d:
            if isinstance(d[k], dict):
                d[k] = _DumbNamespace(d[k])
            if isinstance(d[k], list):
                d[k] = [_DumbNamespace(d[k][i]) if isinstance(d[k][i], dict) else d[k][i] for i in range(len(d[k]))]
        self.__dict__.update(d)

    def keys(self):
        return list(self.__dict__.keys())

    def __getitem__(self, item):
        return self.__dict__[item]

    def as_dict(self):
        return self._d


def construct_epoch_dset(file, config):
    raw = mne.io.read_raw_fif(file, preload = config.preload)
    
    # Exclude channel index by pattern match
    if config.rename_channels:
        picks = config.picks
        renaming_map = dict()
        new_picks = []
        for idx in [idx for idx in range(len(raw.ch_names)) if raw.ch_names[idx] in picks]:
            for new_ch in config.rename_channels.keys():
                if config.rename_channels[new_ch] == raw.ch_names[idx]:
                    renaming_map[raw.ch_names[idx]] = new_ch
                    new_picks.append(idx)
        raw = raw.rename_channels(renaming_map)

    if config.normalize and config.preload:
        raw.apply_function(lambda x: (x-np.mean(x))/np.std(x))
    sfreq = raw.info['sfreq']
    new_sfreq = 256

    event_map = {v: v for v in config.events.values()}
    events = mne.events_from_annotations(raw, event_id=config.events, chunk_duration=eval(config.chunk_duration))[0]
    
    epochs = mne.Epochs(raw, events, tmin=config.tmin, tmax=config.tmin + config.tlen - 1 / sfreq, preload=config.preload, decim=1,
                        baseline=config.baseline, reject_by_annotation=config.drop_bad)
    recording = EpochTorchRecording(epochs, ch_ind_picks=new_picks, event_mapping=event_map,
                                    force_label=True)
    if config.deep1010:
        _dum = _DumbNamespace(dict(channels=recording.channels, info=dict(data_max=config.data_max,
                                                                            data_min=config.data_min)))
        xform = MappingDeep1010(_dum)
        recording.add_transform(xform)

    if config.upsample:
        recording.add_transform(TemporalInterpolation(config.tlen*new_sfreq, new_sfreq=new_sfreq))
    
    return recording



class EEG_dataset(TorchDataset):
    def __init__(self, dn3_dset, augmentation_config, preloaded = False, fine_tune_mode = False, standardize_epochs = False, bendr_setup = False):
        super().__init__()
        self.dn3_dset = dn3_dset
        self.aug_config = augmentation_config
        self.preloaded = preloaded
        self.fine_tune_mode = fine_tune_mode
        self.standardize_epochs = standardize_epochs
        self.bendr_setup = bendr_setup

    def __len__(self):
        return len(self.dn3_dset)
    
    
    def __getitem__(self, index):
        signal, label = self.dn3_dset.__getitem__(index)

        if self.standardize_epochs:
            if self.standardize_epochs == 'total':
                signal = (signal-torch.mean(signal))/torch.std(signal)
            elif self.standardize_epochs == 'channelwise':
                signal = (signal-torch.mean(signal, axis = 1)[:,np.newaxis])/torch.std(signal, axis = 1)[:,np.newaxis]

        if self.bendr_setup == 1:
            sig = torch.zeros([6, signal.shape[1]])
            sig[0:2,:] = signal[0,:]
            sig[4:,:] = signal[1,:]
            signal = sig
        elif self.bendr_setup == 2:
            sig = torch.zeros([6, signal.shape[1]])
            sig[0,:] = signal[0,:]
            sig[-1,:] = signal[1,:]
            signal = sig
        elif self.bendr_setup == 3:
            sig = torch.zeros([6, signal.shape[1]])
            sig[1,:] = signal[0,:]
            sig[-2,:] = signal[1,:]
            signal = sig
    
        return signal, label
    

class SeqCLR_dataset(TorchDataset):
    def __init__(self, 
                 dn3_dset, 
                 fine_tune_mode = False, 
                 window_length = 3000,
                 standardize_epochs = False):
        super().__init__()
        self.dn3_dset = dn3_dset
        self.fine_tune_mode = fine_tune_mode
        self.standardize_epochs = standardize_epochs
        self.window_length = window_length*100

    def __len__(self):
        return len(self.dn3_dset)
    
    def __getitem__(self, index):
        signal, label = self.dn3_dset.__getitem__(index)

        if not self.fine_tune_mode:
            # select two random channels
            ch1, ch2 = np.random.choice(np.arange(signal.shape[0]), 2, replace = False)
            signal = signal[ch1] - signal[ch2]
            # select two random augmentations
            aug1, aug2 = np.random.choice(np.arange(6), 2, replace = False)
            signal_1 = SeqCLR_augmentations(signal, aug1, window_length=self.window_length)
            signal_2 = SeqCLR_augmentations(signal, aug2, window_length=self.window_length)
            return signal_1.unsqueeze(1), signal_2.unsqueeze(1)
        else:
            return signal.transpose(0,1)*10**6, label
        

def SeqCLR_augmentations(x, aug_selection, window_length):
    padding = int((len(x)-window_length)/2)
    if aug_selection == 0:
        # time shift
        max_shift = int(window_length/80)
        shift = np.random.randint(1, max_shift)
        sign = np.random.choice([-1, 1])
        x = x[padding + shift * sign: padding + shift * sign + window_length]
    elif aug_selection == 1:
        # amplitude scaling
        scale = np.random.uniform(0.5, 2)
        x = x[padding:padding+window_length] * scale
    elif aug_selection == 2:
        # DC shift
        shift = np.random.uniform(-10, 10)
        x = x[padding:padding+window_length] + shift
    elif aug_selection == 3:
        # zero_masking
        max_mask = int(window_length*0.0375)
        mask_length = np.random.randint(1, max_mask)
        mask_start = np.random.randint(0, window_length - mask_length)
        x = x[padding:padding+window_length]
        x[mask_start:mask_start+mask_length] = 0
    elif aug_selection == 4:
        # Gaussian noise
        sigma = np.random.uniform(0, 0.2)
        x = x[padding:padding+window_length] + np.random.normal(0, sigma)
    elif aug_selection == 5:
        # band stop filter
        x = x[padding:padding+window_length]
        fs = 100
        center = np.random.uniform(2.8, 41.3)
        width = 5
        low = (center - width/2) 
        high = (center + width/2)
        sos = butter(5, [low, high], btype='bandstop', fs=fs, output='sos')
        x = sosfiltfilt(sos, x)
        x = torch.tensor(x.copy())
    return x
