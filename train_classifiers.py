import os
import tqdm
from argparse import ArgumentParser
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import models.utils as utils
import time

import datasets.celeba as celeba

def main(hparams, writer):
    device = torch.device(hparams.device)# if torch.cuda.is_available() else "cpu")

    print(device)
    input_shape = (hparams.image_width, hparams.image_height)

    discriminator_secret = utils.get_discriminator_model(
            name=hparams.classifier_name,
            num_classes=2,
            pretrained=False,
            device=device)

    print("Number of  parameters: {}".format(utils.count_parameters(discriminator_secret)))

    #discriminator_secret = models.resnet50(pretrained=True)
    #num_ftrs = discriminator_secret.fc.in_features
    #discriminator_secret.fc = nn.Linear(num_ftrs, 2)
    #discriminator_secret.to(device)
    secret_loss_function = nn.CrossEntropyLoss()
    secret_optimizer = optim.Adam(discriminator_secret.parameters(), lr=0.0002, betas=(0.5, 0.99))

    discriminator_utility = utils.get_discriminator_model(
            name=hparams.classifier_name, 
            num_classes=2,
            pretrained=False,
            device=device)
    #discriminator_utility = models.resnet50(pretrained=True)
    #num_ftrs = discriminator_utility.fc.in_features
    #discriminator_utility.fc = nn.Linear(num_ftrs, 2)
    #discriminator_utility.to(device)
    utility_loss_function = nn.CrossEntropyLoss()
    utility_optimizer = optim.Adam(discriminator_utility.parameters(), lr=0.0002, betas=(0.5, 0.99))

    celeba_traindataset = celeba.CelebADataset(
            split='train',
            in_memory=True, #True,
            input_shape=input_shape,
            utility_attr=hparams.utility_attr,
            secret_attr=hparams.secret_attr,
            transform=transforms.Compose([
                celeba.ToTensor(),
            ]))

    celeba_validdataset = celeba.CelebADataset(
            split='valid',
            in_memory=True,
            input_shape=input_shape,
            utility_attr=hparams.utility_attr,
            secret_attr=hparams.secret_attr,
            transform=transforms.Compose([
                celeba.ToTensor(),
            ]))

    trainloader = DataLoader(celeba_traindataset, batch_size=hparams.batch_size, shuffle=True, num_workers=8)
    validloader = DataLoader(celeba_validdataset, batch_size=hparams.batch_size, num_workers=8)

    best_utility_val_loss = np.inf
    best_secret_val_loss = np.inf
    for epoch in range(hparams.max_epochs):
        # TRAINING
        #utility_running_loss = 0.0
        #secret_running_loss = 0.0
        for i, batch in tqdm.tqdm(enumerate(trainloader, 0)):
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
            #secret_running_loss += secret_loss.item()
            #utility_running_loss += utility_loss.item()
            #if i % 100 == 99:    # print every 100 mini-batches
            #    print('[%d, %5d] utility loss: %.3f secret loss: %.3f' %
            #        (epoch + 1, i + 1, utility_running_loss / 100, secret_running_loss / 100))
            #    utility_running_loss = 0.0
            #    secret_running_loss = 0.0

        # VALIDATION
        utility_acc = 0
        secret_acc = 0
        utility_loss = 0
        secret_loss = 0
        for i, batch in enumerate(validloader, 0):
            images  = batch['image'].to(device)
            utility = batch['utility'].to(device)
            secret  = batch['secret'].to(device)
            pred_secret  = discriminator_secret(images)
            pred_utility  = discriminator_utility(images)

            u_loss = utility_loss_function(pred_utility, torch.squeeze(utility))
            s_loss = utility_loss_function(pred_secret, torch.squeeze(secret))

            def accuracy(pred, true):
                u   = true.cpu().numpy().flatten()
                p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
                acc = np.sum(u == p)/len(u)

                return acc

            utility_loss += u_loss.item()
            secret_loss += s_loss.item()
            secret_acc += accuracy(pred_secret, secret)
            utility_acc += accuracy(pred_utility, utility)

        utility_loss = utility_loss / (i+1)
        secret_loss = secret_loss / (i+1)

        utility_acc = utility_acc / (i+1)
        secret_acc = secret_acc / (i+1)
        print('Epoch: %d utility acc : %.3f secret acc: %.3f' %
                (epoch + 1, utility_acc, secret_acc))
        print('Epoch: %d utility loss : %.3f secret loss: %.3f' %
                (epoch + 1, utility_loss, secret_loss))
        writer.add_scalar('acc/utility', utility_acc)
        writer.add_scalar('acc/secret', secret_acc)
        writer.add_scalar('loss/utility', utility_loss)
        writer.add_scalar('loss/secret', secret_loss)

        experiment_path = os.path.join('artifacts', hparams.experiment_name)
        # serialize the best models
        if utility_loss < best_utility_val_loss:
            print("utility was better, saving model ...")
            torch.save(discriminator_utility.state_dict(), "{}/classifier_utility_{}x{}.h5".format(experiment_path, hparams.image_width, hparams.image_height))
            best_utility_val_loss = utility_loss
        if secret_loss < best_secret_val_loss:
            print("secret was better, saving model ...")
            torch.save(discriminator_secret.state_dict(), "{}/classifier_secret_{}x{}.h5".format(experiment_path, hparams.image_width, hparams.image_height))
            best_secret_val_loss = secret_loss

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--classifier_name', type=str, default='resnet18')
    parser.add_argument('--image_width', type=int, default=64)
    parser.add_argument('--image_height', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--utility_attr', type=str, default='Male')
    parser.add_argument('--secret_attr', type=str, default='Smiling')
    hparams = parser.parse_args()

    experiment_path = os.path.join('artifacts', hparams.experiment_name)
    print(experiment_path)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    writer = SummaryWriter(log_dir = experiment_path)

    main(hparams, writer)
