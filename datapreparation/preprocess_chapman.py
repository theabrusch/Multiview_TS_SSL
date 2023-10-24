import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

def check_filedata(filedata):
    if filedata.shape[-1] != 5000:
        return False
    if np.isnan(filedata).any():
        return False
    return True

base_path = '/Users/theb/Desktop/data/chapman/'
out_path = '/Users/theb/Desktop/data/chapman'
ECGfiles = glob(base_path + 'ECGDataDenoised/*.csv')
database = pd.read_excel('/Users/theb/Desktop/data/chapman/Diagnostics.xlsx')
database.index = database['FileName']

# regroup labels
old_rhythms = ['AF','SVT','ST','AT','AVNRT','AVRT','SAAWR','SI','SA']
new_rhythms = ['AFIB','GSVT','GSVT','GSVT','GSVT','GSVT','GSVT','SR','SR']
database['Rhythm'] = database['Rhythm'].replace(old_rhythms,new_rhythms)
labelencoder = LabelEncoder()
database['Rhythm'] = labelencoder.fit_transform(database['Rhythm'])

# divide ECGfiles into train (60%), test (20%) and val (20%)
train, test = train_test_split(ECGfiles, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.25, random_state=42)

splits = {'train': train, 'val': val, 'test': test}
broken_files = {}
for split in splits.keys():
    data = []
    labels = []
    broken_files[split] = []
    for file in splits[split]:
        file_data = pd.read_csv(file, header=None).values
        file_data = np.expand_dims(file_data.T,0)
        if not check_filedata(file_data):
            broken_files[split].append(file)
            continue
        data.append(file_data)
        file_name = file.split('/')[-1].split('.')[0]
        label = database.loc[file_name]['Rhythm']
        labels.append(label)
    data = torch.Tensor(np.concatenate(data))
    labels = torch.Tensor(labels)
    split_data = {'samples': data, 'labels': labels}
    print('N samples in', split, 'set:', len(split_data['samples']))
    print('Label balance in', split, 'set:', split_data['labels'].unique(return_counts=True))
    torch.save(split_data, f'{out_path}/{split}.pt')

print('Broken files:', broken_files)
