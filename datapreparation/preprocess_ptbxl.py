import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import ast
import wfdb

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = '/Users/theb/Desktop/data/ptbxl/'
sampling_rate = 500
fold_to_use = 10

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic).map(lambda x: x[0] if len(x) else np.nan)

# since we are only using the dataset for pretraining, we save info on a training set and a validation set
train = Y[Y.strat_fold != fold_to_use]
val = Y[Y.strat_fold == fold_to_use]

# we only need the diagnostic superclass, filenames and patient ids
train = train[['diagnostic_superclass', 'filename_lr', 'filename_hr', 'patient_id']]
val = val[['diagnostic_superclass', 'filename_lr', 'filename_hr', 'patient_id']]

train_data = load_raw_data(train, sampling_rate, path)
print('Done loading train data')
val_data = load_raw_data(val, sampling_rate, path)
print('Done loading val data')
le = LabelEncoder()
train_labels = le.fit_transform(train.diagnostic_superclass)
val_labels = le.transform(val.diagnostic_superclass)

train = {'samples': torch.Tensor(train_data).transpose(1,2), 'labels': torch.Tensor(train_labels)}
val = {'samples': torch.Tensor(val_data).transpose(1,2), 'labels': torch.Tensor(val_labels)}

output_path = '/Users/theb/Desktop/data/ptbxl/'
torch.save(train, output_path+'train.pt')
torch.save(val, output_path+'val.pt')
