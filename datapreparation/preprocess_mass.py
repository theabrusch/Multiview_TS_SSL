import mne
import os
import glob
import argparse


def preprocess_EEG(path, out_folder = None):

    raw = mne.io.read_raw_edf(path, preload = True)
    file_split = path.split('/')[-1].split(' ')[0]
    subject = file_split.split('-')[-1]
    session = '-'.join(file_split.split('-')[:2])
    file_path = '/'.join(path.split('/')[:-1])
        
    anno_path = f'{file_path}/{session}-{subject} Base.edf'
    
    annot_train = mne.read_annotations(anno_path)
    raw.set_annotations(annot_train, emit_warning=False)
    raw = raw.resample(100)
    channel_picks = [ch for ch in raw.info['ch_names'] if ch.split(' ')[0] == 'EEG']
    raw = raw.pick(channel_picks)
    out_path = f'{out_folder}{subject}/'
    os.makedirs(out_path, exist_ok = True)
    raw.save(f'{out_path}{subject}{session}_raw.fif', overwrite = True)

def main(args):
    root_folder = args.root_folder + '* PSG.edf'
    subjects = glob.glob(root_folder)

    for i, subject in enumerate(subjects):
        print('Subject', i+1, 'of', len(subjects))
        preprocess_EEG(subject, out_folder = args.out_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, default='/dtu-compute/mass-dataset/SS3/')
    parser.add_argument('--out_folder', type=str, default='/path/to/save/fif/')
    args = parser.parse_args()
    main(args)