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
    device = "cuda:{}".format(torch.cuda.current_device())
    #device = torch.device(hparams.device)# if torch.cuda.is_available() else "cpu")

    print(device)
    input_shape = (hparams.image_width, hparams.image_height)

    discriminator = utils.get_discriminator_model(
            name=hparams.classifier_name,
            num_classes=2,
            pretrained=False,
            device=device)

    print("Number of  parameters: {}".format(utils.count_parameters(discriminator)))

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.99))

    celeba_traindataset = celeba.CelebADataset(
            split='train',
            in_memory=False, #True,
            input_shape=input_shape,
            utility_attr='Male',
            secret_attr=hparams.attr,
            transform=transforms.Compose([
                celeba.ToTensor(),
            ]))

    celeba_validdataset = celeba.CelebADataset(
            split='valid',
            in_memory=False,
            input_shape=input_shape,
            utility_attr='Male',
            secret_attr=hparams.attr,
            transform=transforms.Compose([
                celeba.ToTensor(),
            ]))

    trainloader = DataLoader(celeba_traindataset, batch_size=hparams.batch_size, shuffle=True, num_workers=8)
    validloader = DataLoader(celeba_validdataset, batch_size=hparams.batch_size, num_workers=8)

    best_val_loss = np.inf
    for epoch in range(hparams.max_epochs):
        # TRAINING
        for i, batch in tqdm.tqdm(enumerate(trainloader, 0)):
            images  = batch['image'].to(device)
            secret  = batch['secret'].to(device)

            # update discriminator
            discriminator.zero_grad()
            pred = discriminator(images)
            loss = loss_function(pred, torch.squeeze(secret))
            loss.backward()
            optimizer.step()

        # VALIDATION
        #acc = 0
        loss = 0
        for i, batch in enumerate(validloader, 0):
            images  = batch['image'].to(device)
            secret  = batch['secret'].to(device)

            pred    = discriminator(images)

            s_loss = loss_function(pred, torch.squeeze(secret))

#            def accuracy(pred, true):
#                u   = true.cpu().numpy().flatten()
#                p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
#                acc = np.sum(u == p)/len(u)
#
#                return acc

            loss += s_loss.item()
#            acc  += accuracy(pred, secret)

        loss = loss / (i+1)
#        acc  = acc / (i+1)

#        print('Epoch: %d utility acc : %.3f' % (epoch + 1, acc))
        print('Epoch: %d utility loss : %.3f' % (epoch + 1, loss))
#        writer.add_scalar('acc', acc, epoch)
        writer.add_scalar('loss', loss, epoch)

        experiment_path = os.path.join('artifacts', hparams.experiment_name)
        # serialize the best models
        if loss < best_val_loss:
            print("loss was better, saving model ...")
            torch.save(discriminator.state_dict(), "{}/classifier_{}x{}.h5".format(experiment_path, hparams.image_width, hparams.image_height))
            best_val_loss = loss

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier_name', type=str, default='resnet18')
    parser.add_argument('--image_width', type=int, default=64)
    parser.add_argument('--image_height', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--attr', type=str, default='Male')
    hparams = parser.parse_args()

    experiment_path = os.path.join('artifacts', hparams.experiment_name)
    print(experiment_path)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    writer = SummaryWriter(log_dir = experiment_path)

    main(hparams, writer)
