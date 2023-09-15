from scipy.io import loadmat
import h5py
import numpy as np
import glob
import scipy.signal
import os
import mne
import argparse

def import_signal_names(file_name):
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])
        n_samples = int(s[0][3])
        Fs        = int(s[0][2])

        s = s[1:-1]
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples

def extract_labels(path):
    data = h5py.File(path, 'r')
    length = data['data']['sleep_stages']['wake'].shape[1]
    labels = np.zeros((length, 6)) 

    for i, label in enumerate(data['data']['sleep_stages'].keys()):
        labels[:,i] = data['data']['sleep_stages'][label][:]
    
    return labels, list(data['data']['sleep_stages'].keys())

def resample_signal(data, labels, old_fs):
    diff = np.diff(labels, axis = 0)
    cutoff = np.where(diff[:,4] != 0)[0]+1
    data, labels = data[cutoff[0]+1:,:], labels[cutoff[0]+1:,:]

    new_fs = 100
    num = int(len(data)/(old_fs/new_fs))
    resampled_data = scipy.signal.resample(data, num = num, axis = 0)
    resampled_labels = labels[::int((old_fs/new_fs)),:]
    return resampled_data.astype(np.int16), resampled_labels.astype(np.int16)


def preprocess_EEG(folder,remove_files = False, out_folder = None):
    files = glob.glob(f'{folder}/*')
    data = None
    labels = None
    Fs = None
    for file in files:
        if '.hea' in file:
            s, Fs, n_samples = import_signal_names(file)
            if remove_files:
                os.remove(file)
        elif '-arousal.mat' in file:
            labels, label_names = extract_labels(file)
            if remove_files:
                os.remove(file)
        elif 'mat' in file:
            data = loadmat(file)['val'][:6, :]
            if remove_files:
                os.remove(file)

    if not data is None:
        diff = np.diff(labels, axis = 0)
        cutoff = np.where(diff[:,4] != 0)[0]+1
        data, labels = data[:, cutoff[0]+1:], labels[cutoff[0]+1:,:]

        info = mne.create_info(s[:6], Fs, ch_types = 'eeg')
        mne_dataset = mne.io.RawArray(data, info)

        events = process_labels_to_events(labels, label_names)
        label_dict = dict(zip(np.arange(0,6), label_names))
        events = np.array(events)
        event_dict = dict(zip(label_names, np.arange(0,6)))
        f = lambda x: label_dict[x]
        annotations = mne.Annotations(onset = events[:,0]/Fs, duration = events[:,1]/Fs, description  = list(map(f,events[:,2])))
        mne_dataset.set_annotations(annotations)

        mne_dataset.resample(sfreq = 100)
        epoch_events = mne.events_from_annotations(mne_dataset, chunk_duration = 30)
        info = mne.create_info(['STI'], mne_dataset.info['sfreq'], ['stim'])
        stim_data = np.zeros((1, len(mne_dataset.times)))
        stim_raw = mne.io.RawArray(stim_data, info)
        mne_dataset.add_channels([stim_raw], force_update_info=True)
        mne_dataset.add_events(epoch_events[0], stim_channel = 'STI')
        mne_dataset.save(f'{out_folder}/001_30s_raw.fif', overwrite = True)

def relocate_EEG_data(folder, remove_files = True):

    data_file = mne.read_epochs(f'{folder}/001_30s.fif')
    #h5py.File(f'{folder}/data.hdf5', 'r')
    new_name = f'{folder}/001_30s_epo.fif'
    data_file.save(new_name)
    if remove_files:
        os.remove(f'{folder}/data.mat')
        os.remove(f'{folder}/001_30s.fif')

def process_labels_to_events(labels, label_names):
    new_labels = np.argmax(labels, axis = 1)
    lab = new_labels[0]
    events = []
    start = 0
    i = 0
    while i < len(new_labels)-1:
        while new_labels[i] == lab and i < len(new_labels)-1:
            i+=1
        end = i
        dur = end +1 - start
        events.append([start, dur, lab])
        lab = new_labels[i]
        start = i+1
    return events




out_folder = '/Users/theb/Desktop/physionet.org/physionet.org/files/challenge-2018/1.0.0/training/'
out_folder = '/Users/theb/Desktop/training_raw/'
root_folder = '/Volumes/SED/training/'

def main(args):
    subjects = os.listdir(root_folder)
    for i, subject in enumerate(subjects):
        print('Processing subject', i+1, 'of', len(subjects))
        subj_folder = os.path.join(root_folder, subject)
        subj_out_folder = os.path.join(out_folder, subject)
        if not os.path.exists(subj_out_folder):
            os.makedirs(subj_out_folder, exist_ok = True)
        try:
            preprocess_EEG(subj_folder, out_folder = subj_out_folder, remove_files = False)
        except:
            print('Issue with subject', subject)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, default='/path/to/physionet.org/physionet.org/files/challenge-2018/1.0.0/training/')
    parser.add_argument('--out_folder', type=str, default='/path/to/save/fif/')
    args = parser.parse_args()
    main(args)