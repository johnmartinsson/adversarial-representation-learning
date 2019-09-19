import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

import datasets.celeba as celeba
from models.unet import UNet

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(device)
    input_shape = (224, 224)

    discriminator_secret = models.resnet50(pretrained=True)
    num_ftrs = discriminator_secret.fc.in_features
    discriminator_secret.fc = nn.Linear(num_ftrs, 2)
    discriminator_secret.to(device)
    discriminator_secret.load_state_dict(torch.load('artifacts/classifiers/discriminator_secret_5.h5'))

    discriminator_utility = models.resnet50(pretrained=True)
    num_ftrs = discriminator_utility.fc.in_features
    discriminator_utility.fc = nn.Linear(num_ftrs, 2)
    discriminator_utility.to(device)
    discriminator_utility.load_state_dict(torch.load('artifacts/classifiers/discriminator_utility_5.h5'))

    noise_dim = 10
    G1 = UNet(3, 3, image_width=input_shape[0], image_height=input_shape[1], noise_dim = noise_dim)
    G1.to(device)
    G1.load_state_dict(torch.load('artifacts/wse/adversarial_G1_35.h5'))

    G2 = UNet(3, 3, image_width=input_shape[0], image_height=input_shape[1], noise_dim = noise_dim)
    G2.to(device)
    G2.load_state_dict(torch.load('artifacts/wse/adversarial_G2_35.h5'))

    celeba_validdataset = celeba.CelebADataset(
            ann_file='data/validation_annotations.csv',
            image_dir='data/imgs',
            transform=transforms.Compose([
                celeba.Rescale(input_shape),
                celeba.ToTensor(),
            ]))

    validloader = DataLoader(celeba_validdataset, batch_size=8, shuffle=True, num_workers=8)

    # VALIDATION
    utility_acc = 0
    secret_acc = 0
    utility_acc_2 = 0
    secret_acc_2 = 0
    for i, batch in enumerate(validloader, 0):
        images  = batch['image'].to(device)
        utility = batch['utility'].to(device)
        secret  = batch['secret'].to(device)

        z1 = torch.randn(len(images), noise_dim).to(device)
        z2 = torch.randn(len(images), noise_dim).to(device)

        images_1 = G1(images, z1)
        images_2 = G2(images_1, z2)

        pred_secret  = discriminator_secret(images)
        pred_utility  = discriminator_utility(images)

        pred_secret_2  = discriminator_secret(images_2)
        pred_utility_2  = discriminator_utility(images_2)

        def accuracy(pred, true):
            u   = true.cpu().numpy().flatten()
            p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
            acc = np.sum(u == p)/len(u)

            return acc

        secret_acc += accuracy(pred_secret, secret)
        utility_acc += accuracy(pred_utility, utility)
        secret_acc_2 += accuracy(pred_secret_2, secret)
        utility_acc_2 += accuracy(pred_utility_2, utility)
    print('----------------------------------------------')
    print(' Uncensored images ')
    print('----------------------------------------------')
    print('utility acc : %.3f secret acc: %.3f' %
            (utility_acc / (i+1), secret_acc / (i+1)))
    print('')
    print('----------------------------------------------')
    print(' Censored images ')
    print('----------------------------------------------')
    print('utility acc : %.3f secret acc: %.3f' %
            (utility_acc_2 / (i+1), secret_acc_2 / (i+1)))
