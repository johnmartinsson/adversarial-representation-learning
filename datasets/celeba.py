from __future__ import print_function, division
import tqdm
import os
import torch
import pandas as pd
import numpy as np
import skimage
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

    def __init__(self, split, transform=None, input_shape=(224, 224), in_memory=False, utility_attr='Male', secret_attr='Smiling', normalize=False):
        """
        Args:
            split (string): which split to load.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if split == 'train':
            ann_file = './data/training_annotations.csv'
        if split == 'valid':
            ann_file = './data/validation_annotations.csv'
        if split == 'test':
            ann_file = './data/test_annotations.csv'

        self.ann_frame = pd.read_csv(ann_file)
        self.image_dir = './data/imgs'
        self.transform = transform
        self.in_memory = in_memory
        self.input_shape = input_shape
        self.utility_attr = utility_attr
        self.secret_attr = secret_attr

        if self.in_memory:
            # check shape of first image and assume same for all
            (width, height) = self.input_shape
            # load images into memory
            print("loading images into memory ...")

            image_data_file = './data/celeba_images_{}_{}x{}.npy'.format(split, width, height)
            if not os.path.exists(image_data_file):
                self.images = np.empty((len(self.ann_frame), width, height, 3))
                for idx in tqdm.tqdm(range(len(self.ann_frame))):
                    img_name = os.path.join(self.image_dir,
                                            self.ann_frame.iloc[idx, 1])
                    image = skimage.io.imread(img_name)
                    image = skimage.transform.resize(image, input_shape)

                    self.images[idx] = image
                np.save(image_data_file, self.images)
            else:
                self.images = np.load(image_data_file)

        if normalize:
            mean = np.mean(self.images, axis=(0,1,2))
            std  = np.std(self.images, axis=(0,1,2))
            self.images = self.images - mean
            self.images = self.images / std
            self.mean = mean
            self.std = std

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
