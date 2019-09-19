import os
import tqdm
from argparse import ArgumentParser
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import torchvision
import torchvision.models as models

from models.unet import UNet

import datasets.celeba as celeba
import loss_functions

import matplotlib.pyplot as plt

training_batch_counter = 0

def run_experiment(hparams, writer):
    hparams.device = torch.device(hparams.device if torch.cuda.is_available() else "cpu")
    artifacts_path = os.path.join('artifacts', hparams.experiment_name)

    beta1          = 0.5
    beta2          = 0.99
    nb_discriminator_steps = 1
    nb_generator_steps     = 1

    image_height   = 224
    image_width    = 224

    channels_in  = 3
    channels_out = 3
    image_dir    = './data/imgs'

    print(hparams.device)
    input_shape = (image_height, image_width)

    # initialize transformation networks
    print("initialize networks ...")
    G1 = UNet(channels_in, channels_out, image_width=image_width, image_height=image_height, noise_dim = hparams.noise_dim, additive_noise=hparams.use_additive_noise, activation='sigmoid')
    G1.to(hparams.device)

    G2 = UNet(channels_in, channels_out, image_width=image_width, image_height=image_height, noise_dim = hparams.noise_dim, additive_noise=hparams.use_additive_noise, activation='sigmoid')
    G2.to(hparams.device)

    if hparams.use_weighted_squared_error:
        G_w = UNet(channels_in, 1, image_width=image_width, image_height=image_height, noise_dim = None, activation='sigmoid')
        G_w.to(hparams.device)
        G_optimizer = optim.Adam(
            list(G1.parameters()) + list(G2.parameters()) + list(G_w.parameters()), lr=hparams.g_lr, betas=(beta1, beta2))
    else:
        G_w = None
        G_optimizer = optim.Adam(
            list(G1.parameters()) + list(G2.parameters()), lr=hparams.g_lr, betas=(beta1, beta2))

    # initialize discriminator
    # secret discriminator
    Ds = models.resnet50(pretrained=True)
    num_ftrs = Ds.fc.in_features
    Ds.fc = nn.Linear(num_ftrs, 2)

    # real/fake discriminator
    if hparams.use_real_fake_discriminator:
        Drf = models.resnet50(pretrained=False)
        num_ftrs = Drf.fc.in_features
        Drf.fc = nn.Linear(num_ftrs, 2)
    else:
        Drf = None

    Ds.to(hparams.device)
    Ds_optimizer = optim.Adam(Ds.parameters(), lr=hparams.ds_lr, betas=(beta1, beta2))

    if not Drf is None:
        Drf.to(hparams.device)
        Drf_optimizer = optim.Adam(Drf.parameters(), lr=hparams.drf_lr, betas=(beta1, beta2))
    else:
        Drf_optimizer = None

    print("loading datasets ...")
    celeba_traindataset = celeba.CelebADataset(
            ann_file='./data/training_annotations.csv',
            image_dir=image_dir,
            transform=transforms.Compose([
                celeba.Rescale(input_shape),
                celeba.ToTensor(),
            ]))

    celeba_validdataset = celeba.CelebADataset(
            ann_file='./data/validation_annotations.csv',
            image_dir=image_dir,
            transform=transforms.Compose([
                celeba.Rescale(input_shape),
                celeba.ToTensor(),
            ]))

    trainloader = DataLoader(celeba_traindataset, batch_size=hparams.batch_size, num_workers=8, shuffle=True)
    validloader = DataLoader(celeba_validdataset, batch_size=hparams.batch_size, num_workers=8)

    # load fixed discriminators
    print("loading fixed discriminators ...")
    Ds_fix = load_fixed_classifier('classifiers/classifier_secret.hdf5')
    Ds_fix.to(hparams.device)
    Du_fix = load_fixed_classifier('classifiers/classifier_utility.hdf5')
    Du_fix.to(hparams.device)

    # TODO
    print("training ...")
    for i_epoch in range(hparams.max_epochs):
        train_indices = np.random.choice(list(range(len(celeba_traindataset))), size=hparams.nb_train_batches*hparams.batch_size, replace=False)
        valid_indices = np.random.choice(list(range(len(celeba_validdataset))), size=hparams.nb_valid_batches*hparams.batch_size, replace=False)
        trainsampler = SubsetRandomSampler(train_indices)
        validsampler = SubsetRandomSampler(valid_indices)
        trainloader = DataLoader(celeba_traindataset, batch_size=hparams.batch_size, num_workers=8, sampler=trainsampler)
        validloader = DataLoader(celeba_validdataset, batch_size=hparams.batch_size, num_workers=8, sampler=validsampler)

        print("training step ...")
        result = fit(G1, G2, G_w, Ds, Drf, G_optimizer, Ds_optimizer, Drf_optimizer, trainloader, hparams, writer)
        print("validation step ...")
        (val_secret_acc, val_utility_acc), val_result = validate(G1, G2, G_w, Ds_fix, Du_fix, validloader, hparams)
        print("secret acc: {}, utility_acc: {}".format(val_secret_acc, val_utility_acc))

        # log stuff
        writer.add_scalar('accuracy/secret_acc', val_secret_acc, i_epoch)
        writer.add_scalar('accuracy/utility_acc', val_utility_acc, i_epoch)

        names = ['real images', 'representations', 'fake_images']
        for i, name in enumerate(names):
            grid = torchvision.utils.make_grid(result[i])
            writer.add_image('train/' + name, grid, 0)

            grid = torchvision.utils.make_grid(val_result[i])
            writer.add_image('valid/' + name, grid, 0)

        if hparams.use_weighted_squared_error:
            grid = torchvision.utils.make_grid(result[3])
            writer.add_image("train/weights", grid, 0)

            grid = torchvision.utils.make_grid(val_result[3])
            writer.add_image("valid/weights", grid, 0)

        # save stuff
        save_models(G1, G2, G_w, Ds, Drf, artifacts_path, i_epoch)

def load_fixed_classifier(weight_path):
    classifier = models.resnet50(pretrained=True)
    num_ftrs = classifier.fc.in_features
    classifier.fc = nn.Linear(num_ftrs, 2)
    classifier.load_state_dict(torch.load(weight_path))

    return classifier

def save_models(G1, G2, G_w, Ds, Drf, artifacts_path, epoch):
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)
    torch.save(G1.state_dict(), os.path.join(artifacts_path, "G1_{}.hdf5".format(epoch)))
    torch.save(G2.state_dict(), os.path.join(artifacts_path, "G2_{}.hdf5".format(epoch)))
    if not G_w is None:
        torch.save(G_w.state_dict(), os.path.join(artifacts_path, "G_w_{}.hdf5".format(epoch)))
    torch.save(Ds.state_dict(), os.path.join(artifacts_path, "Ds_{}.hdf5".format(epoch)))

    if not Drf is None:
        torch.save(Drf.state_dict(), os.path.join(artifacts_path, "Drf_{}.hdf5".format(epoch)))

def fit(G1, G2, G_w, Ds, Drf, G_optimizer, Ds_optimizer, Drf_optimizer, trainloader, hparams, writer):
    global training_batch_counter
    # define loss functions
    secret_loss_function = torch.nn.CrossEntropyLoss()
    real_fake_loss_function = torch.nn.CrossEntropyLoss()

    def weighted_mean_squared_error(x1, x2, w, C, lambd):
        # penalize area larger than C
        penalty = hparams.lambd*F.relu(torch.mean(w)-C)

        writer.add_scalar('loss/weights', torch.mean(w).cpu().detach().numpy(), training_batch_counter)
        
        w = torch.full(w.shape, 1.0).to(hparams.device) - w # broadcast
        return torch.mean(torch.pow(x1-x2, 2) * w) + penalty

    def mean_squared_error(x1, x2):
        return torch.mean(torch.pow(x1-x2, 2))

    entropy_loss_function = loss_functions.HLoss()

    # TRAINING
    for i_batch, batch in tqdm.tqdm(enumerate(trainloader, 0)):
        images  = batch['image'].to(hparams.device)
        utility = batch['utility'].to(hparams.device)
        secret  = batch['secret'].to(hparams.device)

        # draw noise
        z1 = torch.randn(len(images), hparams.noise_dim).to(hparams.device)
        z2 = torch.randn(len(images), hparams.noise_dim).to(hparams.device)

        # train discriminators
        # update secret predictor parameters
        images_1 = G1(images, z1)
        secret_pred = Ds(images_1)
        d_secret_loss = secret_loss_function(secret_pred, secret.view(len(secret_pred)))

        Ds_optimizer.zero_grad()
        d_secret_loss.backward()
        Ds_optimizer.step()

        # update real/fake predictor parameters
        if not Drf is None:
            images_1 = G1(images, z1)
            images_2 = G2(images_1, z1)
            real_pred = Drf(images)
            fake_pred = Drf(images_2)
            real_true = torch.full((len(images),), 1.0).long().to(hparams.device)
            fake_true = torch.full((len(images_2), ), 0.0).long().to(hparams.device)
            rf_pred = torch.cat((real_pred, fake_pred), dim=0)
            rf_true = torch.cat((real_true, fake_true), dim=0)
            d_real_fake_loss = real_fake_loss_function(rf_pred, rf_true)

            Drf_optimizer.zero_grad()
            d_real_fake_loss.backward()
            Drf_optimizer.step()

        # train generators
        images_1 = G1(images, z1)
        secret_pred_1 = Ds(images_1)
        images_2 = G2(images_1, z2)

        if not Drf is None:
            fake_pred = Drf(images_2)
            fake_fool = torch.full((len(images_2),), 1.0).long().to(hparams.device)
            g_real_fake_loss = real_fake_loss_function(fake_pred, fake_fool)

        if hparams.use_weighted_squared_error:
            SE_w = G_w(images)
            rc_loss = weighted_mean_squared_error(images, images_2, SE_w, C=hparams.weight_budget, lambd=hparams.lambd)
        else:
            rc_loss = mean_squared_error(images, images_2)

        secret_loss_g  = entropy_loss_function(secret_pred_1)
        generator_loss = hparams.alpha * rc_loss - hparams.beta * secret_loss_g
        
        if not Drf is None:
            generator_loss = generator_loss + hparams.gamma * g_real_fake_loss

        # update parameters
        G_optimizer.zero_grad()
        generator_loss.backward()
        G_optimizer.step()


        # log stuff
        writer.add_scalar('generator_loss/rc_loss', hparams.alpha * rc_loss.item(), training_batch_counter)
        writer.add_scalar('generator_loss/secret_loss', hparams.beta * secret_loss_g.item(), training_batch_counter)
        if not Drf is None:
            writer.add_scalar('generator_loss/real_fake_loss', hparams.gamma * g_real_fake_loss.item(), training_batch_counter)
            writer.add_scalar('discriminator_loss/real_fake_loss', d_real_fake_loss.item(), training_batch_counter)
        writer.add_scalar('discriminator_loss/secret_loss', d_secret_loss.item(), training_batch_counter)

        training_batch_counter = training_batch_counter + 1

    if hparams.use_weighted_squared_error:
        return images, images_1, images_2, SE_w
    else:
        return images, images_1, images_2

def validate(G1, G2, G_w, Ds_fix, Du_fix, validationloader, hparams):
    
    running_secret_acc = 0.0
    running_utility_acc = 0.0

    for i_batch, batch in tqdm.tqdm(enumerate(validationloader, 0)):
        images  = batch['image'].to(hparams.device)
        utility = batch['utility'].to(hparams.device)
        secret  = batch['secret'].to(hparams.device)

        # draw noise
        z1 = torch.randn(len(images), hparams.noise_dim).to(hparams.device)
        z2 = torch.randn(len(images), hparams.noise_dim).to(hparams.device)

        images_1 = G1(images, z1)
        images_2 = G2(images_1, z2)

        secret_pred_fix = Ds_fix(images_2)
        utility_pred_fix = Du_fix(images_2)

        def accuracy(pred, true):
            u   = true.cpu().numpy().flatten()
            p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
            acc = np.sum(u == p)/len(u)

            return acc

        running_secret_acc  += accuracy(secret_pred_fix, secret)
        running_utility_acc += accuracy(utility_pred_fix, utility)

    secret_acc  = running_secret_acc / len(validationloader)
    utility_acc = running_utility_acc / len(validationloader)

    if not G_w is None:
        se_w = G_w(images)
        return (secret_acc, utility_acc), (images, images_1, images_2, se_w)
    else:
        return (secret_acc, utility_acc), (images, images_1, images_2)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--experiment_name", type=str, default='default')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--nb_train_batches", type=int, default=500)
    parser.add_argument("--nb_valid_batches", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=200)

    parser.add_argument("--g_lr", type=float, default=0.0002)
    parser.add_argument("--ds_lr", type=float, default=0.0002)
    parser.add_argument("--drf_lr", type=float, default=0.0002)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--weight_budget", type=int, default=0.10) #3000)
    parser.add_argument("--lambd", type=float, default=1000.0)

    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)

    parser.add_argument("--use_additive_noise", type=bool, default=False)
    parser.add_argument("--use_weighted_squared_error", type=bool, default=True)
    parser.add_argument("--use_real_fake_discriminator", type=bool, default=False)


    # hyperparameters

    hparams = parser.parse_args()

    writer = SummaryWriter(log_dir = os.path.join('artifacts', hparams.experiment_name))

    run_experiment(hparams, writer)
