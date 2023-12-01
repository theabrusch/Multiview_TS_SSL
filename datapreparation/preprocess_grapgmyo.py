from scipy.io import loadmat
import numpy as np
from scipy.signal import resample
import os

path = '/Users/theb/Desktop/data/capgmyo/dba-preprocessed-001/001-001-001.mat'
data = loadmat(path)
emg_data = data['data']
window_size = 0.2 # 200 ms
sample_frequency = 1000 # 1kHz
overlap = 0.05 # 50 ms
window_length = int(window_size*sample_frequency)
overlap_length = int(overlap*sample_frequency)

# divide data into windows
windows = np.zeros((int((len(emg_data)-window_length)/overlap_length), emg_data.shape[1], window_length))
for i in range(windows.shape[0]):
    windows[i,:,:] = emg_data[i*overlap_length:i*overlap_length+window_length, :].T
labels = np.zeros((windows.shape[0]))
labels[:] = data['gesture']