from scipy.io import loadmat
import numpy as np
from scipy.signal import resample
import os

dataset_path = '/Users/theb/Desktop/data/ninaprodb2/'
window_size = 2 # 1 second
resample_freq = 1000 # 200 Hz
sample_freq = 2000 # 2kHz
output_path = '/Users/theb/Desktop/data/ninaprodb2/processed/' + f'{resample_freq}Hz_{window_size}s/' 
if not os.path.exists(output_path):
    os.makedirs(output_path)

subjects = os.listdir(dataset_path)
for subject in subjects:
    if subject == '.DS_Store':
        continue
    subject_path = dataset_path + subject + '/'
    sessions = os.listdir(subject_path)
    for file in sessions:
        if file.endswith('.mat'):
            data = loadmat(subject_path + file)
            emg_data = data['emg']
            label = data['restimulus']
            # resample data
            # cutoff end point to ensure multiples of sample_freq
            if not resample_freq == sample_freq:
                emg_data = emg_data[:int(len(emg_data)/sample_freq)*sample_freq, :] 
                resampled_emg_data = resample(emg_data, int(len(emg_data)/sample_freq*resample_freq), axis=0)
            else:
                resampled_emg_data = emg_data
            # split into windows
            windows = np.zeros((int(len(emg_data)/sample_freq*window_size), window_size*resample_freq, emg_data.shape[1], ))
            labels = np.zeros((int(len(emg_data)/sample_freq*window_size)))
            for i in range(windows.shape[0]):
                windows[i,:,:] = resampled_emg_data[i*resample_freq:(i+1)*resample_freq, :]
                # get most frequently occuring label in window
                labels[i] = np.argmax(np.bincount(label[i*sample_freq:(i+1)*sample_freq,0]))
            save_path = output_path + file[:-4] + '.npz'
            np.savez_compressed(save_path, windows=windows, labels=labels)
            print('Saved', subject_path + file[:-4] + '.npz')

