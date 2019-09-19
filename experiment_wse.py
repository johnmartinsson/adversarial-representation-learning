import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import torchvision.models as models

from models.unet import UNet

import datasets.celeba as celeba
import loss_functions

import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # settings optimizer
    learning_rate = 0.0002
    beta1 = 0.5
    noise_dim = 10

    print(device)
    input_shape = (224, 224)

    G1 = UNet(3, 3, image_width=input_shape[0], image_height=input_shape[1], noise_dim = 10)
    G1.to(device)

    G2 = UNet(3, 3, image_width=input_shape[0], image_height=input_shape[1], noise_dim = 10)
    G2.to(device)

    G_w = UNet(3, 1, image_width=input_shape[0], image_height=input_shape[1], noise_dim = None)
    G_w.to(device)

    G_optimizer = optim.Adam(
        list(G1.parameters()) + list(G2.parameters()) + list(G_w.parameters()), lr=learning_rate, betas=(beta1, 0.99))

    # load discriminator
    Ds = models.resnet50(pretrained=True)
    num_ftrs = Ds.fc.in_features
    Ds.fc = nn.Linear(num_ftrs, 2)
    Ds.to(device)

    Ds_optimizer = optim.Adam(Ds.parameters(), lr=learning_rate, betas=(beta1, 0.99))

    secret_loss_function = torch.nn.CrossEntropyLoss()

    def reconstruction_loss_function(x1, x2, w, C=3000, lambd=1000):
        # penalize area larger than C
        penalty = lambd*F.relu(torch.sum(w)-C)
        
        w = torch.full(w.shape, 1.0).to(device) - w # broadcast
        return torch.mean(torch.pow(x1-x2, 2) * w) + penalty

    entropy_loss_function = loss_functions.HLoss()

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

    batch_size = 8
    trainloader = DataLoader(celeba_traindataset, batch_size=batch_size, num_workers=8, shuffle=True)
    validloader = DataLoader(celeba_validdataset, batch_size=batch_size, num_workers=8)

    nb_discriminator_steps = 1
    nb_generator_steps = 1

    for epoch in range(50):
        discriminator_losses = []
        generator_losses = []
        # TRAINING
        train_discriminator = True
        i_d = 0
        i_g = 0
        for i_batch, batch in enumerate(trainloader, 0):
            images  = batch['image'].to(device)
            utility = batch['utility'].to(device)
            secret  = batch['secret'].to(device)

            # draw noise
            z1 = torch.randn(len(images), noise_dim).to(device)
            z2 = torch.randn(len(images), noise_dim).to(device)

            if train_discriminator:
                images_1 = G1(images, z1)
                secret_pred = Ds(images_1)
                discriminator_loss = secret_loss_function(secret_pred, torch.squeeze(secret))

                # update parameters
                Ds_optimizer.zero_grad()
                discriminator_loss.backward()
                Ds_optimizer.step()

                i_d += 1

                if i_d == nb_discriminator_steps:
                    #print("batch: {} toggle (discriminator)".format(i))
                    # toggle after x batches
                    train_discriminator = not train_discriminator
                    i_d = 0
            else:
                images_1 = G1(images, z1)
                secret_pred_1 = Ds(images_1)
                images_2 = G2(images_1, z2)
                SE_w = G_w(images)

                rc_loss        = 100 * reconstruction_loss_function(images, images_2, SE_w)
                secret_loss_g  = entropy_loss_function(secret_pred_1)
                generator_loss = rc_loss - secret_loss_g

                # update parameters
                G_optimizer.zero_grad()
                generator_loss.backward()
                G_optimizer.step()

                i_g += 1

                if i_g == nb_generator_steps:
                    #print("batch: {} toggle (generator)".format(i))
                    # toggle after 10 batches
                    train_discriminator = not train_discriminator
                    i_g = 0

            if i_batch % 100 == 99:
                print("######################################")
                print("generator loss: {}".format(generator_loss.item()))
                print("discriminator loss: {}".format(discriminator_loss.item()))
                print("reconstruction loss: {}".format(rc_loss.item()))
                print("entropy loss: {}".format(secret_loss_g.item()))
                discriminator_losses.append(discriminator_loss.item())
                generator_losses.append(generator_loss.item())

                image = batch['image'].to(device)

                image_1 = G1(image, z1)
                image_2 = G2(image_1, z2)
                SE_w    = G_w(image)

                image = image.cpu().detach().numpy().transpose((0,2,3,1))
                image_1 = image_1.cpu().detach().numpy().transpose((0,2,3,1))
                image_2 = image_2.cpu().detach().numpy().transpose((0,2,3,1))
                image_se = SE_w.cpu().detach().numpy().transpose((0,2,3,1))

                fig, axarr = plt.subplots(batch_size, 4, figsize=((3*2, 8*2)))
                for i in range(batch_size):
                    axarr[i, 0].imshow(image[i])
                    axarr[i, 1].imshow(image_1[i])
                    axarr[i, 2].imshow(image_2[i])
                    axarr[i, 3].imshow(np.squeeze(image_se[i]))

                    axarr[i, 0].axis('off')
                    axarr[i, 1].axis('off')
                    axarr[i, 2].axis('off')
                    axarr[i, 3].axis('off')

                plt.savefig("artifacts/wse/images/{}_{}.png".format(epoch, i_batch))
                plt.close(fig)

                fig = plt.figure()
                plt.plot(generator_losses, label='generator')
                plt.plot(discriminator_losses, label='discriminator')
                plt.legend(loc="upper right")
                plt.ylabel("loss")
                plt.savefig("artifacts/losses_epoch_{}.png".format(epoch))
                plt.close(fig)

        print("serialize models ...")
        # serialize the model
        torch.save(G1.state_dict(), "artifacts/wse/adversarial_G1_{}.h5".format(epoch))
        torch.save(G2.state_dict(), "artifacts/wse/adversarial_G2_{}.h5".format(epoch))
        torch.save(Ds.state_dict(), "artifacts/wse/adversarial_Ds_{}.h5".format(epoch))
