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
import json
import random

from models.unet import UNet
from models import utils

import datasets.celeba as celeba
import loss_functions

import matplotlib.pyplot as plt

#training_batch_counter = 0

def run_experiment(hparams, writer):
    hparams.device = torch.device(hparams.device if torch.cuda.is_available() else "cpu")
    experiment_path = os.path.join('artifacts', hparams.experiment_name)

    beta1          = 0.5
    beta2          = 0.99
    #nb_discriminator_steps = 1
    #nb_generator_steps     = 1

    image_height   = hparams.image_height #224
    image_width    = hparams.image_width #224

    #channels_in  = 3
    #channels_out = 3
    #image_dir    = './data/imgs'

    print(hparams.device)
    input_shape = (image_height, image_width)

    # initialize transformation networks
    print("initialize networks ...")
    G1 = utils.get_generator_model(hparams.generator_name, hparams.device, hparams.noise_dim, hparams.use_additive_noise)
    #G1 = UNet(channels_in, channels_out, image_width=image_width, image_height=image_height, noise_dim = hparams.noise_dim, additive_noise=hparams.use_additive_noise, activation='sigmoid')
    #G1.to(hparams.device)
    print("G1: {} parameters".format(utils.count_parameters(G1)))

    G2 = utils.get_generator_model(hparams.generator_name, hparams.device, hparams.noise_dim, hparams.use_additive_noise)
    #G2 = UNet(channels_in, channels_out, image_width=image_width, image_height=image_height, noise_dim = hparams.noise_dim, additive_noise=hparams.use_additive_noise, activation='sigmoid')
    #G2.to(hparams.device)
    print("G2: {} parameters".format(utils.count_parameters(G2)))

    if hparams.use_weighted_squared_error:
        G_w = utils.get_generator_model('unet_weight', hparams.device, None, hparams.use_additive_noise)
        #G_w = UNet(channels_in, 1, image_width=image_width, image_height=image_height, noise_dim = None, activation='sigmoid')
        #G_w.to(hparams.device)
        print("G_w: {} parameters".format(utils.count_parameters(G_w)))
        G_optimizer = optim.Adam(
            list(G1.parameters()) + list(G2.parameters()) + list(G_w.parameters()), lr=hparams.g_lr, betas=(beta1, beta2))
    else:
        G_w = None
        G_optimizer = optim.Adam(
            list(G1.parameters()) + list(G2.parameters()), lr=hparams.g_lr, betas=(beta1, beta2))

    # initialize discriminator
    # secret discriminator
    Ds = utils.get_discriminator_model(hparams.discriminator_name, num_classes=2, pretrained=hparams.pretrained, device=hparams.device)
    print("Ds: {} parameters".format(utils.count_parameters(Ds)))
    #Ds = models.resnet50(pretrained=True)
    #num_ftrs = Ds.fc.in_features
    #Ds.fc = nn.Linear(num_ftrs, 2)

    # real/fake discriminator
    if hparams.use_real_fake_discriminator:
        Drf = utils.get_discriminator_model(hparams.discriminator_name, num_classes=2, pretrained=hparams.pretrained, device=hparams.device)
        print("Drf: {} parameters".format(utils.count_parameters(Drf)))
        #Drf = models.resnet50(pretrained=True)
        #num_ftrs = Drf.fc.in_features
        #Drf.fc = nn.Linear(num_ftrs, 2)
    else:
        Drf = None

    # optimizers
    Ds_optimizer = optim.Adam(Ds.parameters(), lr=hparams.ds_lr, betas=(beta1, beta2))

    if not Drf is None:
        #Drf.to(hparams.device)
        Drf_optimizer = optim.Adam(Drf.parameters(), lr=hparams.drf_lr, betas=(beta1, beta2))
    else:
        Drf_optimizer = None

    print("loading datasets ...")
    celeba_traindataset = celeba.CelebADataset(
            split = 'train', #ann_file='./data/training_annotations.csv',
            #image_dir=image_dir,
            in_memory=True,
            input_shape=input_shape,
            utility_attr=hparams.utility_attr,
            secret_attr=hparams.secret_attr,
            transform=transforms.Compose([
                celeba.ToTensor(),
            ]))

    celeba_validdataset = celeba.CelebADataset(
            split = 'valid', #ann_file='./data/validation_annotations.csv',
            #image_dir=image_dir,
            in_memory=True,
            input_shape=input_shape,
            utility_attr=hparams.utility_attr,
            secret_attr=hparams.secret_attr,
            transform=transforms.Compose([
                celeba.ToTensor(),
            ]))

    trainloader = DataLoader(celeba_traindataset, batch_size=hparams.batch_size, num_workers=10, shuffle=True)
    validloader = DataLoader(celeba_validdataset, batch_size=hparams.batch_size, num_workers=10)

    # load fixed discriminators
    print("loading fixed discriminators ...")
    Ds_fix = utils.get_discriminator_model('resnet_small', num_classes=2, pretrained=False, device=hparams.device, weights_path='./artifacts/fixed_resnet_small/classifier_secret_64x64.h5')
    #Ds_fix = load_fixed_classifier('classifiers/classifier_secret.hdf5', hparams.device)
    #Ds_fix.to(hparams.device)
    Du_fix = utils.get_discriminator_model('resnet_small', num_classes=2, pretrained=False, device=hparams.device, weights_path='./artifacts/fixed_resnet_small/classifier_utility_64x64.h5')
    #Du_fix = load_fixed_classifier('classifiers/classifier_utility.hdf5', hparams.device)
    #Du_fix.to(hparams.device)

    # initialize models with previous training weights

    if hparams.resume_training:
        print("load previous models weights ...")
        G1.load_state_dict(torch.load(os.path.join(experiment_path, 'G1.hdf5')))
        G2.load_state_dict(torch.load(os.path.join(experiment_path, 'G2.hdf5')))
        if G_w is not None:
            G_w.load_state_dict(torch.load(os.path.join(experiment_path, 'G_w.hdf5')))
        Ds.load_state_dict(torch.load(os.path.join(experiment_path, 'Ds.hdf5')))
        if Drf is not None:
            Drf.load_state_dict(torch.load(os.path.join(experiment_path, 'Drf.hdf5')))

    trainloader = DataLoader(celeba_traindataset, batch_size=hparams.batch_size, num_workers=8, shuffle=True)
    validloader = DataLoader(celeba_validdataset, batch_size=hparams.batch_size, num_workers=8)
    # start training
    for i_epoch in range(hparams.max_epochs):

        #train_indices = np.random.choice(list(range(len(celeba_traindataset))), size=hparams.nb_train_batches*hparams.batch_size, replace=False)
        #valid_indices = np.random.choice(list(range(len(celeba_validdataset))), size=hparams.nb_valid_batches*hparams.batch_size, replace=False)
        #trainsampler = SubsetRandomSampler(train_indices)
        #validsampler = SubsetRandomSampler(valid_indices)
        #trainloader = DataLoader(celeba_traindataset, batch_size=hparams.batch_size, num_workers=8, sampler=trainsampler)
        #validloader = DataLoader(celeba_validdataset, batch_size=hparams.batch_size, num_workers=8, sampler=validsampler)

        print("training step ...")
        result = fit(G1, G2, G_w, Ds, Drf, G_optimizer, Ds_optimizer, Drf_optimizer, trainloader, hparams, writer)
        print("validation step ...")
        (val_secret_acc, val_utility_acc), val_result = validate(G1, G2, G_w, Ds_fix, Du_fix, validloader, hparams)
        print("secret acc: {}, utility_acc: {}".format(val_secret_acc, val_utility_acc))

        # log stuff
        writer.add_scalar('accuracy/secret_acc', val_secret_acc, i_epoch)
        writer.add_scalar('accuracy/utility_acc', val_utility_acc, i_epoch)

        # TODO: uncomment to save images
        #names = ['real images', 'representations', 'fake_images']
        #for i, name in enumerate(names):
        #    grid = torchvision.utils.make_grid(val_result[i])
        #    writer.add_image('valid/' + name, grid, 0)

        #if hparams.use_weighted_squared_error:
        #    grid = torchvision.utils.make_grid(val_result[3])
        #    writer.add_image("valid/weights", grid, 0)

        # save stuff
        save_models(G1, G2, G_w, Ds, Drf, experiment_path) #, i_epoch)

#def load_fixed_classifier(weight_path, device):
#    classifier = models.resnet50(pretrained=True)
#    num_ftrs = classifier.fc.in_features
#    classifier.fc = nn.Linear(num_ftrs, 2)
#    classifier.load_state_dict(torch.load(weight_path, map_location=device))
#
#    return classifier

def save_models(G1, G2, G_w, Ds, Drf, experiment_path): #, epoch):
    torch.save(G1.state_dict(), os.path.join(experiment_path, "G1.hdf5"))
    torch.save(G2.state_dict(), os.path.join(experiment_path, "G2.hdf5"))
    if not G_w is None:
        torch.save(G_w.state_dict(), os.path.join(experiment_path, "G_w.hdf5"))
    torch.save(Ds.state_dict(), os.path.join(experiment_path, "Ds.hdf5"))

    if not Drf is None:
        torch.save(Drf.state_dict(), os.path.join(experiment_path, "Drf.hdf5"))

def fit(G1, G2, G_w, Ds, Drf, G_optimizer, Ds_optimizer, Drf_optimizer, trainloader, hparams, writer):
    #global training_batch_counter
    # define loss functions
    secret_loss_function = torch.nn.CrossEntropyLoss()
    real_fake_loss_function = torch.nn.CrossEntropyLoss()

    def weighted_mean_squared_error(x1, x2, w, C, lambd):
        # penalize area larger than C
        #penalty = hparams.lambd*F.relu(torch.mean(w)-C)
        penalty = hparams.lambd*torch.pow(F.relu(torch.mean(w)-C), 2)

        writer.add_scalar('loss/weights', torch.mean(w).cpu().detach().numpy()) #, training_batch_counter)
        
        w = torch.full(w.shape, 1.0).to(hparams.device) - w # broadcast
        return torch.mean(torch.pow(x1-x2, 2) * w) + penalty

    def mean_weighted_absolute_error(x1, x2, w, C, lambd):
        # penalize area larger than C
        penalty = hparams.lambd*torch.pow(F.relu(torch.mean(w)-C), 2)

        writer.add_scalar('loss/weights', torch.mean(w).cpu().detach().numpy()) #, training_batch_counter)
        
        w = torch.full(w.shape, 1.0).to(hparams.device) - w # broadcast
        return torch.mean(torch.abs(x1-x2) * w) + penalty

    def mean_absolute_error(x1, x2):
        return torch.mean(torch.abs(x1-x2))

    def mean_squared_error(x1, x2):
        return torch.mean(torch.pow(x1-x2, 2))

    if hparams.use_entropy_loss:
        entropy_loss_function = loss_functions.HLoss()
    else:
        entropy_loss_function = torch.nn.CrossEntropyLoss()

    # TRAINING
    for i_batch, batch in tqdm.tqdm(enumerate(trainloader, 0)):
        images  = batch['image'].to(hparams.device)
        utility = batch['utility'].to(hparams.device)
        secret  = batch['secret'].to(hparams.device)

        # draw noise
        z1 = torch.randn(len(images), hparams.noise_dim).to(hparams.device)
        z2 = torch.randn(len(images), hparams.noise_dim).to(hparams.device)

        # train generators
        if not Drf is None:
            z2n = torch.randn(len(images), hparams.noise_dim//2).to(hparams.device)
            fake_secret = torch.randint(0, 2, (len(images), 1)).float().to(hparams.device)
            z2l = fake_secret.repeat(1, hparams.noise_dim//2).to(hparams.device)
            z2 = torch.cat((z2n, z2l), dim=1)

        images_1 = G1(images, z1)

        if i_batch % hparams.nb_discriminator_steps == 0:
            secret_pred_1 = Ds(images_1)
            images_2 = G2(images_1, z2)

            if not Drf is None:
                fake_secret_pred = Drf(images_2)
                g_fake_secret_loss = secret_loss_function(fake_secret_pred, fake_secret.long().view(len(fake_secret_pred)))

            if hparams.use_weighted_squared_error:
                SE_w = G_w(images)
                #rc_loss = weighted_mean_squared_error(images, images_2, SE_w, C=hparams.weight_budget, lambd=hparams.lambd)
                rc_loss = mean_weighted_absolute_error(images, images_2, SE_w, C=hparams.weight_budget, lambd=hparams.lambd)
            else:
                #rc_loss = mean_squared_error(images, images_2)
                rc_loss = mean_absolute_error(images, images_2)

            if hparams.use_entropy_loss:
                secret_loss_g = entropy_loss_function(secret_pred_1)
            else:
                #fake_secret = torch.full(secret.shape, 1.0).to(hparams.device).long() - secret
                secret_loss_g = entropy_loss_function(secret_pred_1, secret.view(len(secret_pred_1)))

            generator_loss = hparams.alpha * rc_loss - hparams.beta * secret_loss_g
            
            if not Drf is None:
                generator_loss = generator_loss + hparams.gamma * g_fake_secret_loss

            # update parameters
            G_optimizer.zero_grad()
            generator_loss.backward()
            G_optimizer.step()

        # train discriminators
        # update secret predictor parameters
        #images_1 = G1(images, z1) # TODO: detach?
        secret_pred = Ds(images_1.detach())
        d_secret_loss = secret_loss_function(secret_pred, secret.view(len(secret_pred)))

        Ds_optimizer.zero_grad()
        d_secret_loss.backward()
        Ds_optimizer.step()

        # update real/fake predictor parameters
        if not Drf is None:
            secret_pred = Drf(images)
            d_real_secret_loss = secret_loss_function(secret_pred, secret.view(len(secret_pred)))

            Drf_optimizer.zero_grad()
            d_real_secret_loss.backward()
            Drf_optimizer.step()

        # log stuff
        #writer.add_scalar('generator_loss/rc_loss', hparams.alpha * rc_loss.item(), training_batch_counter)
        writer.add_scalar('generator_loss/rc_loss', rc_loss.item()) #, training_batch_counter)
        #writer.add_scalar('generator_loss/secret_loss', hparams.beta * secret_loss_g.item(), training_batch_counter)
        writer.add_scalar('generator_loss/secret_loss', secret_loss_g.item()) #, training_batch_counter)
        if not Drf is None:
            #writer.add_scalar('generator_loss/fake_secret_loss', hparams.gamma * g_fake_secret_loss.item(), training_batch_counter)
            writer.add_scalar('generator_loss/fake_secret_loss', g_fake_secret_loss.item()) #, training_batch_counter)
            writer.add_scalar('discriminator_loss/real_secret_loss', d_real_secret_loss.item()) #, training_batch_counter)
        writer.add_scalar('discriminator_loss/secret_loss', d_secret_loss.item()) #, training_batch_counter)

        #training_batch_counter = training_batch_counter + 1

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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = ArgumentParser()
    # training
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--experiment_name", type=str, default='default')
    parser.add_argument("--utility_attr", type=str, default='Male')
    parser.add_argument("--secret_attr", type=str, default='Smiling')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=64)
    parser.add_argument("--image_width", type=int, default=64)
    #parser.add_argument("--nb_train_batches", type=int, default=1000)
    #parser.add_argument("--nb_valid_batches", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--resume_training", type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument("--seed", type=int, default=-1)

    # generator
    parser.add_argument("--generator_name", type=str, default='unet_small')
    parser.add_argument("--g_lr", type=float, default=0.0002)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--weight_budget", type=float, default=0.05)
    parser.add_argument("--lambd", type=float, default=1000.0)

    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)

    parser.add_argument("--use_additive_noise", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--use_weighted_squared_error", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--use_real_fake_discriminator", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--use_entropy_loss", type=str2bool, nargs='?', const=True, default=True)

    # discriminator
    parser.add_argument("--ds_lr", type=float, default=0.002)
    parser.add_argument("--drf_lr", type=float, default=0.002)
    parser.add_argument("--discriminator_name", type=str, default='resnet_small')
    parser.add_argument("--pretrained", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--nb_discriminator_steps", type=int, default=10)

    # hyperparameters

    hparams = parser.parse_args()

    if not hparams.seed == -1:
        print("setting random seed to {} ...".format(hparams.seed))
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        random.seed(hparams.seed)

    # save hparams
    experiment_path = os.path.join('artifacts', hparams.experiment_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    with open(os.path.join(experiment_path, "hparams.json"), 'w') as f:
        json.dump(hparams.__dict__, f, indent=2)

    writer = SummaryWriter(log_dir = experiment_path)

    run_experiment(hparams, writer)
