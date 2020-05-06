from __future__ import print_function, division
import glob
import os

import numpy as np
import pandas as pd

import torch
import librosa

def normalize_wav(array):
    # Subtract the mean, and scale to the interval [-1,1]
    array_minusmean = array - array.mean()
    return array_minusmean/np.abs(array_minusmean).max()

class VoxCelebDataset(torch.utils.data.Dataset):
    """VoxCeleb dataset."""

    def __init__(self, root_dir, normalize=False, transform=None):
        """
        Args:
        """
        ann_file = os.path.join(root_dir, 'vox2_meta_no_spaces.csv')
        self.normalize = normalize
        self.ann_frame = pd.read_csv(ann_file).set_index('VoxCeleb2ID')
        self.wav_dir   = os.path.join(root_dir, 'wav_segments')
        self.wav_files = glob.glob(os.path.join(self.wav_dir, '*.wav'))
        self.transform = transform

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_file = self.wav_files[idx]
        f_id = wav_file.split('/')[-1].split('_')[0]

        # need to explicitly tell librosa NOT to resample ...
        wav, samplerate = librosa.load(wav_file, sr=None)

        if self.normalize:
            wav = normalize_wav(wav)

        annotation = self.ann_frame.loc[f_id]['Gender'] == 'm'
        gender = np.array([annotation])

        sample = {'wav': wav, 'gender': gender}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        wav, gender = sample['wav'], sample['gender']

        return {'wav': torch.from_numpy(wav).float(),
                'gender': torch.from_numpy(gender).long()}
