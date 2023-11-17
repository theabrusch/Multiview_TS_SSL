import torch
from torch.utils.data import Dataset as TorchDataset
from src.datasets.datasets import SSL_dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import wfdb
import numpy as np

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def load_raw_data_ptbxl(path, sampling_rate = 500):
    train_df = pd.read_csv('dataset_load_files/ptbxl_train.csv', index_col=0)
    val_df = pd.read_csv('dataset_load_files/ptbxl_val.csv', index_col=0)

    # labels are strings, encode labels with LabelEncoder
    le = LabelEncoder()
    train_data = load_raw_data(train_df, sampling_rate, path)
    train_labels = le.fit_transform(train_df.diagnostic_superclass)
    val_data = load_raw_data(val_df, sampling_rate, path)
    val_labels = le.transform(val_df.diagnostic_superclass)

    train_dset = SSL_dataset(train_data, train_labels)
    val_dset = SSL_dataset(val_data, val_labels)
    return train_dset, val_dset, (train_data.shape[1], train_data.shape[2], len(le.classes_))