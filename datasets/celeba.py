from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#def to_categorical(y, num_classes):
#    """ 1-hot encodes a tensor """
#    return np.eye(num_classes, dtype='uint8')[y]

class CelebADataset(Dataset):
    """CelebADataset dataset."""

    def __init__(self, ann_file, image_dir, transform=None):
        """
        Args:
            ann_file (string): Path to the file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ann_frame = pd.read_csv(ann_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.ann_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir,
                                self.ann_frame.iloc[idx, 1])
        image = io.imread(img_name)
        annotations = self.ann_frame.iloc[idx, 2:]

        utility = np.array([(float(annotations['Male']) + 1)/2])
        secret  = np.array([(float(annotations['Smiling']) + 1)/2])

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

        img = transform.resize(image, (new_h, new_w))

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
