import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

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

    discriminator_utility = models.resnet50(pretrained=True)
    num_ftrs = discriminator_utility.fc.in_features
    discriminator_utility.fc = nn.Linear(num_ftrs, 2)
    discriminator_utility.to(device)
    utility_loss_function = nn.CrossEntropyLoss()
    utility_optimizer = optim.Adam(discriminator_utility.parameters(), lr=0.0002, betas=(0.5, 0.99))

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

    for epoch in range(5):
        # TRAINING
        utility_running_loss = 0.0
        secret_running_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            images  = batch['image'].to(device)
            utility = batch['utility'].to(device)
            secret  = batch['secret'].to(device)

            # update discriminator
            discriminator_secret.zero_grad()
            secret_pred = discriminator_secret(images)
            secret_loss = secret_loss_function(secret_pred, torch.squeeze(secret))
            secret_loss.backward()
            secret_optimizer.step()

            # update discriminator
            discriminator_utility.zero_grad()
            utility_pred = discriminator_utility(images)
            utility_loss = utility_loss_function(utility_pred, torch.squeeze(utility))
            utility_loss.backward()
            utility_optimizer.step()

            # print statistics
            secret_running_loss += secret_loss.item()
            utility_running_loss += utility_loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] utility loss: %.3f secret loss: %.3f' %
                    (epoch + 1, i + 1, utility_running_loss / 100, secret_running_loss / 100))
                utility_running_loss = 0.0
                secret_running_loss = 0.0

        # VALIDATION
        utility_acc = 0
        secret_acc = 0
        for i, batch in enumerate(validloader, 0):
            images  = batch['image'].to(device)
            utility = batch['utility'].to(device)
            secret  = batch['secret'].to(device)
            pred_secret  = discriminator_secret(images)
            pred_utility  = discriminator_utility(images)

            def accuracy(pred, true):
                u   = true.cpu().numpy().flatten()
                p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
                acc = np.sum(u == p)/len(u)

                return acc

            secret_acc += accuracy(pred_secret, secret)
            utility_acc += accuracy(pred_utility, utility)
        print('Epoch: %d utility acc : %.3f secret acc: %.3f' %
                (epoch + 1, utility_acc / (i+1), secret_acc / (i+1)))

        # serialize the models
        torch.save(discriminator_utility.state_dict(), "classifiers/classifier_utility.h5".format(epoch))
        torch.save(discriminator_secret.state_dict(), "classifiers/clasifier_secret.h5".format(epoch))
