import tqdm
import argparse
import os
import numpy as np
import math
import scipy
import json
import pickle
import time

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

###############################################################################
# Hyperparameters
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--eps", type=float, default=0.5, help="distortion budget")
parser.add_argument("--lambd", type=float, default=100000.0, help="squared penalty")
parser.add_argument("--latent_dim", type=int, default=1024, help="dimensionality of the latent space")
parser.add_argument("--embedding_dim", type=int, default=128, help="dimensionality of embedding space")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--log_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--discriminator_update_interval", type=int, default=1, help="how often to update discriminator")
parser.add_argument("--name", type=str, default='default', help="experiment name")
parser.add_argument("--use_real_fake", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--use_entropy_loss", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--use_filter", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--use_cond", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--use_generator_cond", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("--freeze_filter", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--resume_training", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--discriminator_name", type=str, default='resnet_small_discriminator', help="name of discriminator")
parser.add_argument("--artifacts_dir", type=str, default='default', help="directory to put artifacts in")
parser.add_argument("--utility_attr", type=str, default='Male')
parser.add_argument("--secret_attr", type=str, default='Smiling')
opt = parser.parse_args()
print(opt)

if not (opt.use_real_fake or opt.use_filter):
    raise ValueError("use_real_fake or use_filter must be true")

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# summary writer
artifacts_path = opt.artifacts_dir
os.makedirs(artifacts_path, exist_ok=True)
writer = SummaryWriter(artifacts_path)

# fixed classifiers
secret_classifier = utils.get_discriminator_model('resnet18', num_classes=2, pretrained=False, device=device, weights_path='./artifacts/fixed_classifiers/{}/classifier_{}x{}.h5'.format(opt.secret_attr, opt.img_size, opt.img_size))
utility_classifier = utils.get_discriminator_model('resnet18', num_classes=2, pretrained=False, device=device, weights_path='./artifacts/fixed_classifiers/{}/classifier_{}x{}.h5'.format(opt.utility_attr, opt.img_size, opt.img_size))

secret_classifier.eval()
utility_classifier.eval()

adv_secret_classifier = utils.get_discriminator_model('resnet_small', num_classes=2, pretrained=False, device=device)
tmp_secret_classifier = utils.get_discriminator_model('resnet_small', num_classes=2, pretrained=False, device=device)
# inception model
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
inception_v3 = InceptionV3([block_idx])

# Loss functions
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b

adversarial_loss = torch.nn.CrossEntropyLoss() #BCELoss()
entropy_loss = HLoss()
adversarial_rf_loss = torch.nn.CrossEntropyLoss()
distortion_loss = torch.nn.MSELoss()

# Initialize generator and discriminator

filter = UNetFilter(3, 3, image_width=opt.img_size, image_height=opt.img_size, noise_dim=opt.latent_dim, activation='sigmoid', nb_classes=opt.n_classes, embedding_dim=opt.embedding_dim, use_cond=opt.use_cond)
generator = UNetFilter(3, 3, image_width=opt.img_size, image_height=opt.img_size, noise_dim=opt.latent_dim, activation='sigmoid', nb_classes=opt.n_classes, embedding_dim=opt.embedding_dim, use_cond=opt.use_generator_cond)

discriminator = utils.get_discriminator_model('resnet18', num_classes=2, pretrained=True, device=device)

real_fake_classes = 3 if opt.use_generator_cond else 2
if opt.discriminator_name == 'resnet18_discriminator':
    discriminator_rf = utils.get_discriminator_model('resnet18', num_classes=real_fake_classes, pretrained=True, device=device)
elif opt.discriminator_name == 'resnet_small_discriminator':
    discriminator_rf = utils.get_discriminator_model('resnet_small', num_classes=real_fake_classes, pretrained=False, device=device)
else:
    raise ValueError("discriminator not defined: ", opt.discriminator_name)

print("Generator parameters: ", utils.count_parameters(generator))
print("Filter parameters: ", utils.count_parameters(filter))
print("Discriminator secret parameters: ", utils.count_parameters(discriminator))
print("Discriminator real/fake parameters: ", utils.count_parameters(discriminator_rf))

if cuda:
    inception_v3.cuda()
    filter.cuda()
    discriminator.cuda()
    generator.cuda()
    discriminator_rf.cuda()
    adversarial_loss.cuda()
    adversarial_rf_loss.cuda()
    distortion_loss.cuda()
    adv_secret_classifier.cuda()

###############################################################################
# Load the training and validateion/test data
###############################################################################

train_dataloader = torch.utils.data.DataLoader(
    celeba.CelebADataset(
        split='train',
        in_memory=False,
        input_shape=(opt.img_size, opt.img_size),
        utility_attr=opt.utility_attr, #'Male',
        secret_attr=opt.secret_attr, #'Smiling',
        transform=transforms.Compose([
            celeba.ToTensor(),
    ])),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_workers,
    pin_memory=False,
)

if opt.mode == 'evaluate':
    valid_dataloader = torch.utils.data.DataLoader(
        celeba.CelebADataset(
            split='test',
            in_memory=False,
            input_shape=(opt.img_size, opt.img_size),
            utility_attr=opt.utility_attr, #'Male',
            secret_attr=opt.secret_attr, #'Smiling',
            transform=transforms.Compose([
                celeba.ToTensor(),
        ])),
        batch_size=opt.batch_size,
        num_workers=opt.n_workers,
        pin_memory=False,
    )
else:
    valid_dataloader = torch.utils.data.DataLoader(
        celeba.CelebADataset(
            split='valid',
            in_memory=False,
            input_shape=(opt.img_size, opt.img_size),
            utility_attr=opt.utility_attr, #'Male',
            secret_attr=opt.secret_attr, #'Smiling',
            transform=transforms.Compose([
                celeba.ToTensor(),
        ])),
        batch_size=opt.batch_size,
        num_workers=opt.n_workers,
        pin_memory=False,
    )

# Optimizers
optimizer_f = torch.optim.Adam(filter.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_adv = torch.optim.Adam(adv_secret_classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_tmp = torch.optim.Adam(tmp_secret_classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_d_rf = torch.optim.Adam(discriminator_rf.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def train(i_epoch):
    """ Runs one training epoch. """
    # set train mode

    filter.train()
    discriminator.train()
    generator.train()
    discriminator_rf.train()


    for i_batch, batch in tqdm.tqdm(enumerate(train_dataloader)):
        imgs   = batch['image']
        secret = batch['secret'].float()
        secret = secret.view(secret.size(0))

        if cuda:
            imgs = imgs.cuda()
            secret = secret.cuda()

        batch_size = imgs.shape[0]

        # -----------------
        #  Train Filter
        # -----------------
        if opt.use_filter:
            if not opt.freeze_filter:
                optimizer_f.zero_grad()

                # sample noise as filter input
                z = torch.randn(batch_size, opt.latent_dim).cuda()

                # filter a batch of images
                filter_imgs = filter(imgs, z, secret.long())
                pred_secret = discriminator(filter_imgs)

                # loss measures filters's ability to fool the discriminator under constrained distortion
                ones = Variable(FloatTensor(secret.shape).fill_(1.0), requires_grad=False)
                target = ones-secret.float()
                target = target.view(target.size(0)) #, -1)
                if not opt.use_entropy_loss:
                    f_adversary_loss = adversarial_loss(pred_secret, target.long())
                else:
                    f_adversary_loss = -entropy_loss(pred_secret) # negative entropy
                f_distortion_loss = distortion_loss(filter_imgs, imgs)

                f_loss = f_adversary_loss + opt.lambd * torch.pow(torch.relu(f_distortion_loss-opt.eps), 2)

                #if not opt.use_real_fake:
                f_loss.backward()
                optimizer_f.step()

        # ------------------------
        # Train Generator (Real/Fake)
        # ------------------------
        if opt.use_real_fake:
            optimizer_g.zero_grad()
            # sample noise as filter input
            z1 = torch.randn(batch_size, opt.latent_dim).cuda()

            # filter a batch of images
            if opt.use_filter:
                if not opt.freeze_filter:
                    filter_imgs = filter(imgs, z1, secret.long())
                else:
                    with torch.no_grad():
                        filter_imgs = filter(imgs, z1, secret.long())

            else:
                filter_imgs = imgs

            # sample noise as generator input
            z2 = torch.randn(batch_size, opt.latent_dim).cuda()

            # sample secret
            if opt.use_generator_cond:
                gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))
            else:
                gen_secret = Variable(LongTensor(np.random.choice([1.0], batch_size)))

            # generate a batch of images
            gen_imgs = generator(filter_imgs, z2, gen_secret)

            # loss measures generator's ability to fool the discriminator
            pred_secret = discriminator_rf(gen_imgs)
            g_adversary_loss = adversarial_rf_loss(pred_secret, gen_secret)
            g_distortion_loss = distortion_loss(gen_imgs, imgs)

            g_loss = g_adversary_loss + opt.lambd * torch.pow(torch.relu(g_distortion_loss-opt.eps), 2)

            g_loss.backward()
            optimizer_g.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        if opt.use_filter:
            if not opt.freeze_filter:
                optimizer_d.zero_grad()

                pred_secret = discriminator(filter_imgs.detach())
                d_loss = adversarial_loss(pred_secret, secret.long())

                d_loss.backward()
                optimizer_d.step()

        # --------------------------------
        #  Train Discriminator (Real/Fake)
        # --------------------------------

        if i_batch % opt.discriminator_update_interval == 0:
            if opt.use_real_fake:
                optimizer_d_rf.zero_grad()

                #c_imgs = torch.cat((imgs, gen_imgs.detach()), axis=1)
                real_pred_secret = discriminator_rf(imgs)
                fake_pred_secret = discriminator_rf(gen_imgs.detach())

                if opt.use_generator_cond:
                    fake_secret = Variable(LongTensor(fake_pred_secret.size(0)).fill_(2.0), requires_grad=False)
                    d_rf_loss_real = adversarial_rf_loss(real_pred_secret, secret.long())
                    d_rf_loss_fake = adversarial_rf_loss(fake_pred_secret, fake_secret)
                else:
                    fake_secret = Variable(LongTensor(fake_pred_secret.size(0)).fill_(0.0), requires_grad=False)
                    real_secret = Variable(LongTensor(fake_pred_secret.size(0)).fill_(1.0), requires_grad=False)
                    d_rf_loss_real = adversarial_rf_loss(real_pred_secret, real_secret)
                    d_rf_loss_fake = adversarial_rf_loss(fake_pred_secret, fake_secret)

                d_rf_loss = (d_rf_loss_real + d_rf_loss_fake) / 2

                d_rf_loss.backward()
                optimizer_d_rf.step()

        if i_batch % opt.log_interval == 0:
            if opt.use_filter:
                if not opt.freeze_filter:
                    writer.add_scalar('loss/d_loss', d_loss.item(), i_batch + i_epoch*len(train_dataloader))
                    writer.add_scalar('loss/f_loss', f_loss.item(), i_batch + i_epoch*len(train_dataloader))
            if opt.use_real_fake:
                if i_batch % opt.discriminator_update_interval == 0:
                    writer.add_scalar('loss/d_rf_loss', d_rf_loss.item(), i_batch + i_epoch*len(train_dataloader))
                writer.add_scalar('loss/g_loss', g_loss.item(), i_batch + i_epoch*len(train_dataloader))

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
    """ Runs one validation epoch. 

        on validation set if --mode=train
        on test set if       --mode=evaluate
    
    """

    # set eval mode
    filter.eval()
    discriminator.eval()
    generator.eval()
    discriminator_rf.eval()

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
        z = torch.randn(batch_size, opt.latent_dim).cuda()

        if opt.use_filter:
            filter_imgs = filter(imgs, z, secret.long())
        else:
            filter_imgs = imgs

        if opt.use_real_fake:
            z = torch.randn(batch_size, opt.latent_dim).cuda()
            gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))
            filter_imgs = generator(filter_imgs, z, gen_secret)

        secret_adv_pred  = discriminator(filter_imgs)
        secret_pred_fix  = secret_classifier(filter_imgs)
        utility_pred_fix = utility_classifier(filter_imgs)

        def accuracy(pred, true):
            u   = true.cpu().numpy().flatten()
            p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
            acc = np.sum(u == p)/len(u)

            return acc

        running_secret_acc  += accuracy(secret_pred_fix, secret)
        running_secret_adv_acc  += accuracy(secret_adv_pred, secret)
        if opt.use_real_fake:
            running_secret_gen_acc  += accuracy(secret_pred_fix, gen_secret)
        running_utility_acc += accuracy(utility_pred_fix, utility)

        inception_v3.eval()
        a1 = inception_v3(imgs)[0]
        a2 = inception_v3(filter_imgs)[0]
        acts1.append(np.squeeze(a1.detach().cpu().numpy()))
        acts2.append(np.squeeze(a2.detach().cpu().numpy()))

    acts1 = np.concatenate(acts1, axis=0)
    acts2 = np.concatenate(acts2, axis=0)

    fid = compute_frechet_inception_distance(acts1, acts2)

    secret_acc  = running_secret_acc / len(valid_dataloader)
    secret_adv_acc  = running_secret_adv_acc / len(valid_dataloader)
    utility_acc = running_utility_acc / len(valid_dataloader)
    gen_secret_acc = running_secret_gen_acc / len(valid_dataloader)

    return secret_acc, secret_adv_acc, utility_acc, gen_secret_acc, fid, imgs, filter_imgs

def visualize():
    """ Runs through the validation data and visualize the output of the model.
    """

    filter.eval()
    generator.eval()

    for i_batch, batch in tqdm.tqdm(enumerate(valid_dataloader, 0)):
        imgs  = batch['image'].cuda()
        utility = batch['utility'].float().cuda()
        secret  = batch['secret'].float().cuda()
        secret  = secret.view(secret.size(0))
        utility = utility.view(utility.size(0))

        # Sample noise as filter input
        batch_size = imgs.shape[0]

        z = torch.randn(batch_size, opt.latent_dim).cuda()
        z1 = torch.randn(batch_size, opt.latent_dim).cuda()
        z2 = torch.randn(batch_size, opt.latent_dim).cuda()

        if opt.use_filter:
            filter_imgs = filter(imgs, z, secret.long())
        else:
            filter_imgs = imgs

        if opt.use_real_fake:
            gen_secret_0 = Variable(LongTensor(np.random.choice([0.0], batch_size)))
            gen_secret_1 = Variable(LongTensor(np.random.choice([1.0], batch_size)))
            filter_imgs_0 = generator(filter_imgs, z1, gen_secret_0)
            filter_imgs_1 = generator(filter_imgs, z2, gen_secret_1)
        save_dir = os.path.join(artifacts_path, 'visualize')
        utils.save_images_2(imgs, filter_imgs, filter_imgs_0, filter_imgs_1, save_dir, i_batch)

    return

def predict(attr):
    """ Makes predictions for original images and censored images and stores
    the results. """

    preds       = np.zeros((len(valid_dataloader), 2))
    gen_preds   = np.zeros((len(valid_dataloader), 2))
    secrets     = np.zeros((len(valid_dataloader), 1))
    gen_secrets = np.zeros((len(valid_dataloader)))

    classifier = utils.get_discriminator_model('resnet18', num_classes=2, pretrained=False, device=device, weights_path='./artifacts/fixed_classifiers/{}/classifier_{}x{}.h5'.format(attr, opt.img_size, opt.img_size))

    # set eval mode
    classifier.eval()
    generator.eval()
    filter.eval()

    if cuda:
        classifier.cuda()
    # transform images
    for i_batch, batch in tqdm.tqdm(enumerate(valid_dataloader, 0)):
        imgs  = batch['image'].cuda()
        secret  = batch['secret'].float().cuda()

        batch_size  = imgs.shape[0]
        z1 = torch.randn(batch_size, opt.latent_dim).cuda()
        z2 = torch.randn(batch_size, opt.latent_dim).cuda()

        if opt.use_filter:
            filter_imgs = filter(imgs, z1, secret.long())
        else:
            filter_imgs = imgs

        gen_secret  = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))
        gen_imgs    = generator(filter_imgs, z2, gen_secret)

        pred        = classifier(imgs)
        gen_pred    = classifier(gen_imgs)

        preds       = np.concatenate((preds,       pred.detach().cpu().numpy()))
        gen_preds   = np.concatenate((gen_preds,   gen_pred.detach().cpu().numpy()))
        secrets     = np.concatenate((secrets,     secret.cpu().numpy()))
        gen_secrets = np.concatenate((gen_secrets, gen_secret.detach().cpu().numpy()))

    # save transformed images
    file_name = "predict_{}_{}x{}.pkl".format(attr, opt.img_size, opt.img_size)
    with open(os.path.join(opt.artifacts_dir, file_name), 'wb') as f:
        pickle.dump((preds, gen_preds, secrets, gen_secrets), f, pickle.HIGHEST_PROTOCOL)

def train_adversary():
    """ Trains an adversary on data from the data censoring process. """

    def accuracy(pred, true):
        u   = true.cpu().numpy().flatten()
        p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
        acc = np.sum(u == p)/len(u)

        return acc

    tmp_secret_classifier.train()
    adv_secret_classifier.train()
    for i_epoch in range(5):
        for i_batch, batch in tqdm.tqdm(enumerate(train_dataloader, 0)):
            imgs    = batch['image'].cuda()
            utility = batch['utility'].float().cuda()
            secret  = batch['secret'].float().cuda()
            secret  = secret.view(secret.size(0))
            utility = utility.view(utility.size(0))

            batch_size = imgs.shape[0]

            z1 = torch.randn(batch_size, opt.latent_dim).cuda()

            if opt.use_filter:
                filter_imgs = filter(imgs, z1, secret.long())
            else:
                filter_imgs = imgs

            if opt.use_real_fake:
                z2 = torch.randn(batch_size, opt.latent_dim).cuda()
                gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))
                filter_imgs = generator(filter_imgs, z2, gen_secret)

            # train tmp
            optimizer_tmp.zero_grad()
            secret_pred = tmp_secret_classifier(imgs)
            loss = adversarial_loss(secret_pred, secret.long())
            loss.backward()
            optimizer_tmp.step()

            if i_batch % 50 == 0:
                acc = accuracy(secret_pred, secret)
                print("secret_tmp_acc: ", acc)

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
    accs2 = []
    tmp_secret_classifier.eval()
    adv_secret_classifier.eval()
    for i_batch, batch in tqdm.tqdm(enumerate(valid_dataloader, 0)):
        imgs    = batch['image'].cuda()
        secret  = batch['secret'].float().cuda()
        secret  = secret.view(secret.size(0))

        batch_size = imgs.shape[0]

        z1 = torch.randn(batch_size, opt.latent_dim).cuda()
        if opt.use_filter:
            filter_imgs = filter(imgs, z1, secret.long())
        else:
            filter_imgs = imgs

        if opt.use_real_fake:
            z2 = torch.randn(batch_size, opt.latent_dim).cuda()
            gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))
            filter_imgs = generator(filter_imgs, z2, gen_secret)

        secret_pred = adv_secret_classifier(filter_imgs.detach())
        acc = accuracy(secret_pred, secret)
        accs1.append(acc)

        secret_pred = tmp_secret_classifier(imgs)
        acc = accuracy(secret_pred, secret)
        accs2.append(acc)
    acc1 = np.mean(accs1)
    acc2 = np.mean(accs2)
    print("test_secret_adv_acc: ", acc1)
    print("test_secret_tmp_acc: ", acc2)
    return acc1

def predict_images():
    filter.eval()
    generator.eval()

    dataset = celeba.CelebADataset(
        split='valid',
        in_memory=False,
        input_shape=(opt.img_size, opt.img_size),
        utility_attr=opt.utility_attr, #'Male',
        secret_attr=opt.secret_attr, #'Smiling',
        transform=transforms.Compose([
            celeba.ToTensor(),
        ])
    )

    nb_samples = 64

    images   = np.zeros((nb_samples, 3, opt.img_size, opt.img_size))
    images_0 = np.zeros((nb_samples, 3, opt.img_size, opt.img_size))
    images_1 = np.zeros((nb_samples, 3, opt.img_size, opt.img_size))

    for i in range(nb_samples):
        batch   = dataset[-i]
        imgs    = batch['image'].cuda().view(1, 3, opt.img_size, opt.img_size)
        secret  = batch['secret'].float().cuda()
        secret  = secret.view(secret.size(0))

        # Sample noise as filter input
        batch_size = imgs.shape[0]

        z = torch.randn(batch_size, opt.latent_dim).cuda()
        z1 = torch.randn(batch_size, opt.latent_dim).cuda()
        z2 = torch.randn(batch_size, opt.latent_dim).cuda()

        if opt.use_filter:
            filter_imgs = filter(imgs, z, secret.long())
        else:
            filter_imgs = imgs

        if opt.use_real_fake:
            gen_secret_0 = Variable(LongTensor(np.random.choice([0.0], batch_size)))
            gen_secret_1 = Variable(LongTensor(np.random.choice([1.0], batch_size)))
            filter_imgs_0 = generator(filter_imgs, z1, gen_secret_0)
            filter_imgs_1 = generator(filter_imgs, z2, gen_secret_1)

            images[i, :, :, :]   = imgs.cpu().detach().numpy()[0]
            images_0[i, :, :, :] = filter_imgs_0.cpu().detach().numpy()[0]
            images_1[i, :, :, :] = filter_imgs_1.cpu().detach().numpy()[0]

        np.save(os.path.join(artifacts_path, 'images.npy'), images)
        np.save(os.path.join(artifacts_path, 'images_0.npy'), images_0)
        np.save(os.path.join(artifacts_path, 'images_1.npy'), images_1)

    return


if opt.mode == 'train':
    with open(os.path.join(artifacts_path, 'config.json'), 'w') as f:
            json.dump(opt.__dict__, f, indent=2)

    if opt.resume_training:
        generator.load_state_dict(torch.load(os.path.join(artifacts_path, 'generator.hdf5')))
        filter.load_state_dict(torch.load(os.path.join(artifacts_path, 'filter.hdf5')))
        discriminator.load_state_dict(torch.load(os.path.join(artifacts_path, 'discriminator.hdf5')))
        discriminator_rf.load_state_dict(torch.load(os.path.join(artifacts_path, 'discriminator_rf.hdf5')))

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

        with open(os.path.join(artifacts_path, "training.log"), "a") as f:
            print("---------------------------------------------------------\n", file = f)
            print("- Epoch: {}\n".format(i_epoch), file = f)
            print("---------------------------------------------------------\n", file = f)
            print("secret_acc: {}\n".format(secret_acc), file = f)
            print("secret_adv_acc: {}\n".format(secret_adv_acc), file = f)
            print("gen_secret_acc: {}\n".format(gen_secret_acc), file = f)
            print("utility_acc: {}\n".format(utility_acc), file = f)
            print("fid: {}\n".format(fid), file = f)


        # train models
        t1 = time.time()
        train(i_epoch)
        t2 = time.time()

        with open(os.path.join(artifacts_path, "training.log"), "a") as f:
            print("minutes: {}\n".format((t2-t1)/60.0), file = f)
            print("\n", file = f)
            print("\n", file = f)

        # save models
        utils.save_model(filter, os.path.join(artifacts_path, "filter.hdf5"))
        utils.save_model(discriminator, os.path.join(artifacts_path, "discriminator.hdf5"))
        utils.save_model(generator, os.path.join(artifacts_path, "generator.hdf5"))
        utils.save_model(discriminator_rf, os.path.join(artifacts_path, "discriminator_rf.hdf5"))
elif opt.mode == 'evaluate':
    generator.load_state_dict(torch.load(os.path.join(artifacts_path, 'generator.hdf5')))
    filter.load_state_dict(torch.load(os.path.join(artifacts_path, 'filter.hdf5')))

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
elif opt.mode == 'predict':
    filter.load_state_dict(torch.load(os.path.join(artifacts_path, 'filter.hdf5')))
    generator.load_state_dict(torch.load(os.path.join(artifacts_path, 'generator.hdf5')))
    for attr in ['Smiling', 'Male', 'Wearing_Lipstick', 'Young', 'High_Cheekbones', 'Mouth_Slightly_Open', 'Heavy_Makeup']:
        predict(attr)
elif opt.mode == 'predict_images':
    generator.load_state_dict(torch.load(os.path.join(artifacts_path, 'generator.hdf5')))
    filter.load_state_dict(torch.load(os.path.join(artifacts_path, 'filter.hdf5')))
    predict_images()
elif opt.mode == 'visualize':
    generator.load_state_dict(torch.load(os.path.join(artifacts_path, 'generator.hdf5')))
    filter.load_state_dict(torch.load(os.path.join(artifacts_path, 'filter.hdf5')))
    visualize()
else:
    print(opt.mode, " not defined.")
