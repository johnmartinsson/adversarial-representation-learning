from datasets import celeba
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt

def main():
    celeba_sample = celeba.CelebADataset(
        split='train',
        in_memory=True,
        input_shape=(64, 64),
        utility_attr='Male',
        secret_attr='Smiling')

    sample = celeba_sample[0]
    image = sample['image']
    utility = sample['utility']
    secret = sample['secret']

    print("plotting images ... ")
    fig, axarr = plt.subplots(3, 4, figsize=(4*2, 3*2))
    for i in range(3):
        for j in range(4):
            idx = j + (i*4)
            sample = celeba_sample[idx]
            image = sample['image']
            assert(np.min(image) >= 0)
            assert(np.max(image) <=1)
            utility = sample['utility']
            secret = sample['secret']
            axarr[i,j].imshow(sample['image'])
            axarr[i,j].set_title("{}, {}".format(secret, utility))
            axarr[i,j].axis('off')

    fig.suptitle("[smiling, male]")
    plt.savefig("sanity_check.png")

if __name__ == '__main__':
    main()
