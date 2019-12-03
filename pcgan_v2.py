import time
import tqdm
import argparse
import os
import numpy as np
import math
import scipy
import json

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F
import torch

import models.utils as utils
from models.unet import UNetFilter
from models.inception import InceptionV3
from models.filter import Filter
from models.discriminator import Discriminator

import datasets.celeba as celeba

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

device = "cuda:{}".format(torch.cuda.current_device())

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--n_balanced", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--eps", type=float, default=0.5, help="distortion budget")
#parser.add_argument("--alpha", type=float, default=0.1, help="distortion budget")
parser.add_argument("--beta", type=float, default=1.0, help="distortion budget")
parser.add_argument("--lambd", type=float, default=100000.0, help="squared penalty")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--embedding_dim", type=int, default=32, help="dimensionality of embedding space")
parser.add_argument("--encoder_dim", type=int, default=32, help="dimensionality of encoding space")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--log_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--name", type=str, default='default', help="experiment name")
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--use_basic_networks", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--in_memory", type=str2bool, nargs='?', const=True, default=True)
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# summary writer
artifacts_path = 'artifacts/{}_eps_{}_lambd_{}_embedding_dim_{}_lr_{}/'.format(opt.name, opt.eps, opt.lambd, opt.embedding_dim, opt.lr)
os.makedirs(artifacts_path, exist_ok=True)
writer = SummaryWriter(artifacts_path)

# fixed classifiers
secret_classifier = utils.get_discriminator_model('resnet_small', num_classes=2, pretrained=False, device=device, weights_path='./artifacts/fixed_resnet_small/classifier_secret_{}x{}.h5'.format(opt.img_size, opt.img_size))
utility_classifier = utils.get_discriminator_model('resnet_small', num_classes=2, pretrained=False, device=device, weights_path='./artifacts/fixed_resnet_small/classifier_utility_{}x{}.h5'.format(opt.img_size, opt.img_size))

#adv_secret_classifier = utils.get_discriminator_model('resnet_small', num_classes=2, pretrained=False, device=device)
# inception model

#block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
#inception_v3 = InceptionV3([block_idx])

# Loss functions
adversarial_loss = torch.nn.CrossEntropyLoss() #BCELoss()
adversarial_rf_loss = torch.nn.CrossEntropyLoss()
distortion_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
if opt.use_basic_networks:
    generator = Filter(input_shape=(3, opt.img_size, opt.img_size), nb_classes=2, encoder_dim=opt.encoder_dim, embedding_dim=opt.embedding_dim, latent_dim=opt.latent_dim)
    discriminator = Discriminator(input_shape=(3, opt.img_size, opt.img_size), out_dim=3, activation=None)
else:
    generator = UNetFilter(3, 3, image_width=opt.img_size, image_height=opt.img_size, noise_dim=opt.latent_dim, activation='sigmoid', nb_classes=opt.n_classes, embedding_dim=opt.embedding_dim)
    #discriminator = utils.get_discriminator_model('resnet18', num_classes=3, pretrained=True, device=device)
    discriminator = Discriminator(input_shape=(3, opt.img_size, opt.img_size), out_dim=3, activation=None)

print("Generator parameters: ", utils.count_parameters(generator))
print("Discriminator real/fake parameters: ", utils.count_parameters(discriminator))

if cuda:
    #inception_v3.cuda()
    discriminator.cuda()
    generator.cuda()
    adversarial_loss.cuda()
    distortion_loss.cuda()
    #adv_secret_classifier.cuda()

train_dataloader = torch.utils.data.DataLoader(
    celeba.CelebADataset(
        split='train',
        in_memory=opt.in_memory,
        input_shape=(opt.img_size, opt.img_size),
        utility_attr='Male',
        secret_attr='Smiling',
        transform=transforms.Compose([
            celeba.ToTensor(),
    ])),
    batch_size=opt.batch_size,
    shuffle=True,
)

if opt.mode == 'evaluate':
    valid_dataloader = torch.utils.data.DataLoader(
        celeba.CelebADataset(
            split='test',
            in_memory=opt.in_memory,
            input_shape=(opt.img_size, opt.img_size),
            utility_attr='Male',
            secret_attr='Smiling',
            transform=transforms.Compose([
                celeba.ToTensor(),
        ])),
        batch_size=opt.batch_size,
    )
else:
    valid_dataloader = torch.utils.data.DataLoader(
        celeba.CelebADataset(
            split='valid',
            in_memory=opt.in_memory,
            input_shape=(opt.img_size, opt.img_size),
            utility_attr='Male',
            secret_attr='Smiling',
            transform=transforms.Compose([
                celeba.ToTensor(),
        ])),
        batch_size=opt.batch_size,
    )

# Optimizers
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#optimizer_adv = torch.optim.Adam(adv_secret_classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def train_adversary():
    def accuracy(pred, true):
        u   = true.cpu().numpy().flatten()
        p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
        acc = np.sum(u == p)/len(u)

        return acc

    for i_epoch in range(5):
        for i_batch, batch in tqdm.tqdm(enumerate(train_dataloader, 0)):
            imgs    = batch['image'].cuda()
            utility = batch['utility'].float().cuda()
            secret  = batch['secret'].float().cuda()
            secret  = secret.view(secret.size(0))
            utility = utility.view(utility.size(0))

            batch_size = imgs.shape[0]

            z1 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))
            filter_imgs = generator(imgs, z1, gen_secret)

            # train adversary
            optimizer_adv.zero_grad()
            secret_pred = adv_secret_classifier(filter_imgs.detach())
            loss = adversarial_loss(secret_pred, secret.long())
            loss.backward()
            optimizer_adv.step()

            if i_batch % 50 == 0:
                acc = accuracy(secret_pred, secret)
                print("secret_adv_acc: ", acc)

    utils.save_model(adv_secret_classifier, os.path.join(artifacts_path, "adv_secret_classifier.hdf5"))

    accs1 = []
    for i_batch, batch in tqdm.tqdm(enumerate(valid_dataloader, 0)):
        imgs    = batch['image'].cuda()
        secret  = batch['secret'].float().cuda()
        secret  = secret.view(secret.size(0))

        batch_size = imgs.shape[0]

        z1 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))
        filter_imgs = generator(imgs, z1, gen_secret)

        secret_pred = adv_secret_classifier(filter_imgs.detach())
        acc = accuracy(secret_pred, secret)
        accs1.append(acc)

    acc1 = np.mean(accs1)
    print("test_secret_adv_acc: ", acc1)
    return acc1

def visualize():
    for i_batch, batch in tqdm.tqdm(enumerate(valid_dataloader, 0)):
        imgs  = batch['image'].cuda()
        utility = batch['utility'].float().cuda()
        secret  = batch['secret'].float().cuda()
        secret  = secret.view(secret.size(0))
        utility = utility.view(utility.size(0))

        # Sample noise as filter input
        batch_size = imgs.shape[0]

        z1 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        z2 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        gen_secret_0 = Variable(LongTensor(np.random.choice([0.0], batch_size)))
        gen_secret_1 = Variable(LongTensor(np.random.choice([1.0], batch_size)))
        filter_imgs_0 = generator(imgs, z1, gen_secret_0)
        filter_imgs_1 = generator(imgs, z2, gen_secret_1)
        save_dir = os.path.join(artifacts_path, 'visualize')
        utils.save_images_2(imgs, filter_imgs, filter_imgs_0, filter_imgs_1, save_dir, i_batch)

    return

def compute_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def compute_frechet_inception_distance(acts1, acts2):
    mu1, sigma1 = compute_activation_statistics(acts1)
    mu2, sigma2 = compute_activation_statistics(acts2)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def validate():
    running_secret_acc = 0.0
    running_secret_adv_acc = 0.0
    running_secret_gen_acc = 0.0
    running_utility_acc = 0.0

    acts1 = []
    acts2 = []

    for i_batch, batch in tqdm.tqdm(enumerate(valid_dataloader, 0)):
        imgs  = batch['image'].cuda()
        utility = batch['utility'].float().cuda()
        secret  = batch['secret'].float().cuda()
        secret  = secret.view(secret.size(0))
        utility = utility.view(utility.size(0))

        # Sample noise as filter input
        batch_size = imgs.shape[0]
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))

        #ones = Variable(LongTensor(secret.shape).fill_(1.0), requires_grad=False)
        #secret_c = ones-secret
        #z1 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        #z2 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        #gen_imgs_1 = generator(imgs, z1, secret)
        #gen_imgs_2 = generator(imgs, z2, secret_c)

        filter_imgs = generator(imgs, z, gen_secret)
        fake_secret = Variable(LongTensor(filter_imgs.size(0)).fill_(2.0), requires_grad=False)

        secret_adv_pred  = discriminator(filter_imgs)
        secret_pred_fix  = secret_classifier(filter_imgs)
        utility_pred_fix = utility_classifier(filter_imgs)

        def accuracy(pred, true):
            u   = true.cpu().numpy().flatten()
            p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
            acc = np.sum(u == p)/len(u)

            return acc

        running_secret_acc  += accuracy(secret_pred_fix, secret)
        running_secret_adv_acc  += accuracy(secret_adv_pred, fake_secret)
        running_secret_gen_acc  += accuracy(secret_pred_fix, gen_secret)
        running_utility_acc += accuracy(utility_pred_fix, utility)

        #inception_v3.eval()
        #a1 = inception_v3(imgs)[0]
        #a2 = inception_v3(filter_imgs)[0]
        #acts1.append(np.squeeze(a1.detach().cpu().numpy()))
        #acts2.append(np.squeeze(a2.detach().cpu().numpy()))

    #acts1 = np.concatenate(acts1, axis=0)
    #acts2 = np.concatenate(acts2, axis=0)

    #fid = compute_frechet_inception_distance(acts1, acts2)
    fid = 0

    secret_acc  = running_secret_acc / len(valid_dataloader)
    secret_adv_acc  = running_secret_adv_acc / len(valid_dataloader)
    utility_acc = running_utility_acc / len(valid_dataloader)
    gen_secret_acc = running_secret_gen_acc / len(valid_dataloader)

    return secret_acc, secret_adv_acc, utility_acc, gen_secret_acc, fid, imgs, filter_imgs
    #return secret_acc, secret_adv_acc, utility_acc, gen_secret_acc, fid, imgs, gen_imgs_1, gen_imgs_2

def train_without_filter(i_epoch):
    #opt.beta = np.maximum(1-i_epoch*0.5/(opt.n_epochs-opt.n_balanced), 0.5)
    print("beta: ", opt.beta)

    for i_batch, batch in tqdm.tqdm(enumerate(train_dataloader)):
        imgs   = batch['image']
        secret = batch['secret'].long()
        secret = secret.view(secret.size(0))

        if cuda:
            imgs = imgs.cuda()
            secret = secret.cuda()

        batch_size = imgs.shape[0]

        # ---------------------------
        # Train Generator (Real/Fake)
        # ---------------------------
        optimizer_g.zero_grad()

        # sample noise as generator input
        z1 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        z2 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # sample secret
        ones = Variable(LongTensor(secret.shape).fill_(1.0), requires_grad=False)
        secret_c = ones-secret
        #gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))

        # generate a batch of images
        #gen_imgs_1 = generator(imgs, z1, secret)
        gen_imgs_2 = generator(imgs, z2, secret_c)

        # loss measures generator's ability to fool the discriminator
        #pred_secret_1       = discriminator(gen_imgs_1)
        pred_secret_2       = discriminator(gen_imgs_2)
        #g_adversary_loss_1  = adversarial_rf_loss(pred_secret_1, secret)
        g_adversary_loss_2  = adversarial_rf_loss(pred_secret_2, secret_c)
        #g_distortion_loss_1 = distortion_loss(gen_imgs_1, imgs)
        g_distortion_loss_2 = distortion_loss(gen_imgs_2, imgs)

        g_adversary_loss  = g_adversary_loss_2
        g_distortion      = g_distortion_loss_2
        g_distortion_loss = (i_epoch * opt.lambd) * torch.pow(torch.relu(g_distortion-opt.eps), 2)

        #g_adversary_loss  = (1-opt.beta) * g_adversary_loss_1  + opt.beta * g_adversary_loss_2
        #g_distortion      = (1-opt.beta) * g_distortion_loss_1 + opt.beta * g_distortion_loss_2
        #g_distortion_loss = opt.lambd * torch.pow(torch.relu(g_distortion-opt.eps), 2)

        g_loss = g_adversary_loss + g_distortion_loss
        g_loss.backward()
        optimizer_g.step()

        # --------------------------------
        #  Train Discriminator (Real/Fake)
        # --------------------------------

        optimizer_d.zero_grad()

        real_pred_secret = discriminator(imgs)
        #fake_pred_secret_1 = discriminator(gen_imgs_1.detach())
        fake_pred_secret_2 = discriminator(gen_imgs_2.detach())

        fake_secret = Variable(LongTensor(fake_pred_secret_2.size(0)).fill_(2.0), requires_grad=False)
        d_loss_real = adversarial_rf_loss(real_pred_secret, secret.long())
        #d_loss_fake_1 = adversarial_rf_loss(fake_pred_secret_1, fake_secret)
        d_loss_fake_2 = adversarial_rf_loss(fake_pred_secret_2, fake_secret)

        #d_loss_fake = (1-opt.beta) * d_loss_fake_1 + opt.beta * d_loss_fake_2
        d_loss_fake = d_loss_fake_2

        d_loss = (d_loss_real + d_loss_fake) / 2

        d_loss.backward()
        optimizer_d.step()

        if i_batch % opt.log_interval:
            writer.add_scalar('loss/g_distortion_loss', g_distortion_loss.item(), i_batch + i_epoch*len(train_dataloader))
            writer.add_scalar('loss/d_loss', d_loss.item(), i_batch + i_epoch*len(train_dataloader))
            writer.add_scalar('loss/g_loss', g_loss.item(), i_batch + i_epoch*len(train_dataloader))


# ----------
#  Training
# ----------

if opt.mode == 'train':
    for i_epoch in range(opt.n_epochs):
        # validate models
        secret_acc, secret_adv_acc, utility_acc, gen_secret_acc, fid, imgs, filter_imgs = validate()
        print("secret_acc: ", secret_acc)
        print("secret_adv_acc: ", secret_adv_acc)
        print("gen_secret_acc: ", gen_secret_acc)
        print("utility_acc: ", utility_acc)
        print("fid: ", fid)

        # log results
        writer.add_scalar('valid/secret_acc', secret_acc, i_epoch)
        writer.add_scalar('valid/secret_adv_acc', secret_adv_acc, i_epoch)
        writer.add_scalar('valid/utility_acc', utility_acc, i_epoch)
        writer.add_scalar('valid/gen_secret_acc', gen_secret_acc, i_epoch)
        writer.add_scalar('valid/fid', fid, i_epoch)
        utils.save_images(imgs, filter_imgs, artifacts_path, i_epoch)

        # train models
        train_without_filter(i_epoch)

        # save models
        utils.save_model(generator, os.path.join(artifacts_path, "generator.hdf5"))
        utils.save_model(discriminator, os.path.join(artifacts_path, "discriminator.hdf5"))
elif opt.mode == 'evaluate':
    generator.load_state_dict(torch.load(os.path.join(artifacts_path, 'generator.hdf5')))

    secret_acc, _, utility_acc, gen_secret_acc, fid, imgs, filter_imgs = validate()
    print("secret_acc: ", secret_acc)
    print("utility_acc: ", utility_acc)
    print("gen_secret_acc: ", gen_secret_acc)
    print("fid: ", fid)
    adv_acc = train_adversary()
    print("secret_adv_acc: ", adv_acc)
    results = {
        'fix_secret_acc' : secret_acc,
        'fix_utility_acc' : utility_acc,
        'secret_adv_acc' : adv_acc,
        'gen_secret_acc' : gen_secret_acc,
        'fid' : fid,
    }
    with open(os.path.join(artifacts_path, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

elif opt.mode == 'visualize':
    generator.load_state_dict(torch.load(os.path.join(artifacts_path, 'generator.hdf5')))
    visualize()
else:
    print(opt.mode, " not defined.")
