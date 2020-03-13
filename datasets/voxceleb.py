from __future__ import print_function, division
import tqdm
import os
import torch
import pandas as pd
import numpy as np
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class VoxCelebDataset(Dataset):
    """VoxCeleb dataset."""

    def __init__(self, split='train', normalize=False):
        """
        Args:
        """
        if split == 'train':
            ann_file = './data/voxceleb/training_annotations.csv'
        if split == 'valid':
            ann_file = './data/voxceleb/validation_annotations.csv'
        if split == 'test':
            ann_file = './data/voxceleb/test_annotations.csv'

        self.ann_frame = pd.read_csv(ann_file)
        self.wav_dir = './data/voxceleb/wav'

    def __len__(self):
        return len(self.ann_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.in_memory:
            image = self.images[idx]
        else:
            img_name = os.path.join(self.image_dir,
                                    self.ann_frame.iloc[idx, 1])
            image = skimage.io.imread(img_name)
            image = skimage.transform.resize(image, self.input_shape)

        annotations = self.ann_frame.iloc[idx, 2:]

        utility = np.array([(float(annotations[self.utility_attr]) + 1)/2])
        secret  = np.array([(float(annotations[self.secret_attr]) + 1)/2])

        sample = {'image': image, 'utility': utility, 'secret':secret}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = skimage.transform.resize(image, (new_h, new_w))

        utility = sample['utility']
        secret  = sample['secret']

        return {'image': img, 'utility': utility, 'secret': secret}

class Normalize(object):
    """Normalize image object from range [0, 1] to range [-1, 1] """
    def __call__(self, sample):
        image, utility, secret = sample['image'], sample['utility'], sample['secret']
        image = (image - 0.5) * 2 # [0, 1] to [-1, 1]
        return {'image': image, 'utility': utility, 'secret': secret}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, utility, secret = sample['image'], sample['utility'], sample['secret']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'utility': torch.from_numpy(utility).long(),
                'secret': torch.from_numpy(secret).long()}
