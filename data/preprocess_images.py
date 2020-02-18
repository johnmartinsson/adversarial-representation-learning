import sys
sys.path.append('./')
import models.utils as utils
import datasets.celeba as celeba
from torchvision import transforms

def main():
    input_shape=(64, 64)
    celeba_traindataset = celeba.CelebADataset(
            split='train',
            in_memory=True, #True,
            input_shape=input_shape,
            utility_attr='Male',
            secret_attr='Smiling',
            transform=transforms.Compose([
                celeba.ToTensor(),
            ]))

    celeba_validdataset = celeba.CelebADataset(
            split='valid',
            in_memory=True,
            input_shape=input_shape,
            utility_attr='Male',
            secret_attr='Smiling',
            transform=transforms.Compose([
                celeba.ToTensor(),
            ]))

    celeba_validdataset = celeba.CelebADataset(
            split='test',
            in_memory=True,
            input_shape=input_shape,
            utility_attr='Male',
            secret_attr='Smiling',
            transform=transforms.Compose([
                celeba.ToTensor(),
            ]))

if __name__ == '__main__':
    main()
