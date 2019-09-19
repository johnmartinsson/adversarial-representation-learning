import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from models.unet import UNet

import datasets.celeba as celeba

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(device)
    input_shape = (224, 224)

    discriminator_secret = models.resnet50(pretrained=True)
    num_ftrs = discriminator_secret.fc.in_features
    discriminator_secret.fc = nn.Linear(num_ftrs, 2)
    discriminator_secret.to(device)
    secret_loss_function = nn.CrossEntropyLoss()
    secret_optimizer = optim.Adam(discriminator_secret.parameters(), lr=0.0002, betas=(0.5, 0.99))

    noise_dim = 10
    G1 = UNet(3, 3, image_width=input_shape[0], image_height=input_shape[1], noise_dim = noise_dim)
    G1.to(device)
    G1.load_state_dict(torch.load('artifacts/wse/adversarial_G1_35.h5'))

    G2 = UNet(3, 3, image_width=input_shape[0], image_height=input_shape[1], noise_dim = noise_dim)
    G2.to(device)
    G2.load_state_dict(torch.load('artifacts/wse/adversarial_G2_35.h5'))

    celeba_traindataset = celeba.CelebADataset(
            ann_file='data/training_annotations.csv',
            image_dir='data/imgs',
            transform=transforms.Compose([
                celeba.Rescale(input_shape),
                celeba.ToTensor(),
            ]))

    celeba_validdataset = celeba.CelebADataset(
            ann_file='data/validation_annotations.csv',
            image_dir='data/imgs',
            transform=transforms.Compose([
                celeba.Rescale(input_shape),
                celeba.ToTensor(),
            ]))

    trainloader = DataLoader(celeba_traindataset, batch_size=8, shuffle=True, num_workers=8)
    validloader = DataLoader(celeba_validdataset, batch_size=8, shuffle=True, num_workers=8)

    for epoch in range(20):
        # TRAINING
        secret_running_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            images  = batch['image'].to(device)
            utility = batch['utility'].to(device)
            secret  = batch['secret'].to(device)

            z1 = torch.randn(len(images), noise_dim).to(device)
            z2 = torch.randn(len(images), noise_dim).to(device)

            images_1 = G1(images, z1)
            images_2 = G2(images_1, z2)

            # update discriminator
            discriminator_secret.zero_grad()
            secret_pred = discriminator_secret(images_2)
            secret_loss = secret_loss_function(secret_pred, torch.squeeze(secret))
            secret_loss.backward()
            secret_optimizer.step()

            # print statistics
            secret_running_loss += secret_loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] secret loss: %.3f' %
                    (epoch + 1, i + 1, secret_running_loss / 100))
                secret_running_loss = 0.0

        # VALIDATION
        secret_acc = 0
        for i, batch in enumerate(validloader, 0):
            images  = batch['image'].to(device)
            utility = batch['utility'].to(device)
            secret  = batch['secret'].to(device)

            z1 = torch.randn(len(images), noise_dim).to(device)
            z2 = torch.randn(len(images), noise_dim).to(device)

            images_1 = G1(images, z1)
            images_2 = G2(images_1, z2)

            pred_secret  = discriminator_secret(images_2)

            def accuracy(pred, true):
                u   = true.cpu().numpy().flatten()
                p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
                acc = np.sum(u == p)/len(u)

                return acc

            secret_acc += accuracy(pred_secret, secret)
        print('Epoch: %d secret acc: %.3f' %
                (epoch + 1, secret_acc / (i+1)))

        # serialize the models
        torch.save(discriminator_secret.state_dict(), "artifacts/classifiers/adversary_discriminator_secret_{}.h5".format(epoch))
