import tqdm
import argparse
import os
import numpy as np
import math

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
from models.encoder import Encoder
from models.decoder import Decoder
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


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--eps", type=float, default=0.5, help="distortion budget")
parser.add_argument("--lambd", type=float, default=1000.0, help="squared penalty")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--encoder_dim", type=int, default=128, help="dimensionality of the representation space")
parser.add_argument("--embedding_dim", type=int, default=32, help="dimensionality of embedding space")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--log_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--name", type=str, default='default', help="experiment name")
parser.add_argument("--use_real_fake", type=str2bool, nargs='?', const=True, default=False)
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# summary writer
artifacts_path = 'artifacts/{}_eps_{}_lambd_{}_embedding_dim_{}/'.format(opt.name, opt.eps, opt.lambd, opt.embedding_dim)
os.makedirs(artifacts_path, exist_ok=True)
writer = SummaryWriter(artifacts_path)

# fixed classifiers
secret_classifier = utils.get_discriminator_model('resnet_small', num_classes=2, pretrained=False, device='cuda:0', weights_path='./artifacts/fixed_resnet_small/classifier_secret_{}x{}.h5'.format(opt.img_size, opt.img_size))
utility_classifier = utils.get_discriminator_model('resnet_small', num_classes=2, pretrained=False, device='cuda:0', weights_path='./artifacts/fixed_resnet_small/classifier_utility_{}x{}.h5'.format(opt.img_size, opt.img_size))

#def save_model(model, path):
#    torch.save(model.state_dict(), path)

#class Encoder(nn.Module):
#    def __init__(self):
#        super(Encoder, self).__init__()
#
#        def block(in_feat, out_feat, dropout_rate):
#            return [
#                nn.Linear(in_feat, out_feat),
#                nn.Dropout(dropout_rate),
#                nn.LeakyReLU(0.2, inplace=True)
#            ]
#
#        self.model = nn.Sequential(
#            nn.Linear(int(np.prod(img_shape)), 512),
#            nn.LeakyReLU(0.2, inplace=True),
#            *block(512, 512, 0.1),
#            *block(512, 512, 0.1),
#            *block(512, 256, 0.1),
#            *block(256, opt.encoder_dim, 0.1),
#        )
#
#    def forward(self, img): #, labels):
#        # Concatenate label embedding and image to produce input
#        return self.model(img)

#class Decoder(nn.Module):
#    def __init__(self):
#        super(Decoder, self).__init__()
#
#        def block(in_feat, out_feat, normalize=True):
#            layers = [nn.Linear(in_feat, out_feat)]
#            if normalize:
#                layers.append(nn.BatchNorm1d(out_feat, 0.8))
#            layers.append(nn.LeakyReLU(0.2, inplace=True))
#            return layers
#
#        self.model = nn.Sequential(
#            *block(opt.latent_dim + opt.encoder_dim + opt.embedding_dim, 128, normalize=False),
#            *block(128, 256),
#            *block(256, 512),
#            *block(512, 1024),
#            nn.Linear(1024, int(np.prod(img_shape))),
#        )
#
#    def forward(self, encoding):
#        return self.model(encoding)

#class Filter(nn.Module):
#    def __init__(self):
#        super(Filter, self).__init__()
#
#        self.label_emb = nn.Embedding(opt.n_classes, opt.embedding_dim)
#        self.encoder = Encoder()
#        self.decoder = Decoder()
#
#    def forward(self, img, noise, label):
#        # Concatenate label embedding and image to produce input
#        img = img.view(img.size(0), -1)
#        encoding = self.encoder(img)
#        encoding = torch.cat((encoding, self.label_emb(label.long()), noise), -1)
#        diff_img = self.decoder(encoding)
#        img = torch.nn.functional.sigmoid(img + diff_img)
#        img = img.view(img.size(0), *img_shape)
#        return img

#class Discriminator(nn.Module):
#    def __init__(self, out_dim=1, activation='sigmoid'):
#        super(Discriminator, self).__init__()
#        self.activation = activation
#
#        self.model = nn.Sequential(
#            nn.Linear(int(np.prod(img_shape)), 512),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Linear(512, 512),
#            nn.Dropout(0.4),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Linear(512, 512),
#            nn.Dropout(0.4),
#            nn.LeakyReLU(0.2, inplace=True),
#            nn.Linear(512, out_dim),
#            #nn.Sigmoid(),
#        )
#
#    def forward(self, img):
#        x = self.model(img.view(img.size(0), -1))
#        if self.activation == 'sigmoid':
#            x = nn.functional.sigmoid(x)
#        return x


# Loss functions
adversarial_loss = torch.nn.BCELoss()
adversarial_rf_loss = torch.nn.CrossEntropyLoss()
distortion_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
filter = Filter()
discriminator = Discriminator()

generator = Filter()
discriminator_rf = Discriminator(out_dim=3, activation=None)

if cuda:
    filter.cuda()
    discriminator.cuda()
    generator.cuda()
    discriminator_rf.cuda()
    adversarial_loss.cuda()
    distortion_loss.cuda()

train_dataloader = torch.utils.data.DataLoader(
    celeba.CelebADataset(
        split='train',
        in_memory=True,
        input_shape=(opt.img_size, opt.img_size),
        utility_attr='Male',
        secret_attr='Smiling',
        transform=transforms.Compose([
            celeba.ToTensor(),
    ])),
    batch_size=opt.batch_size,
    shuffle=True,
)

valid_dataloader = torch.utils.data.DataLoader(
    celeba.CelebADataset(
        split='valid',
        in_memory=True,
        input_shape=(opt.img_size, opt.img_size),
        utility_attr='Male',
        secret_attr='Smiling',
        transform=transforms.Compose([
            celeba.ToTensor(),
    ])),
    batch_size=opt.batch_size,
)

# Optimizers
optimizer_f = torch.optim.Adam(filter.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_d_rf = torch.optim.Adam(discriminator_rf.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def validate():
    running_secret_acc = 0.0
    running_utility_acc = 0.0

    for i_batch, batch in tqdm.tqdm(enumerate(valid_dataloader, 0)):
        imgs  = batch['image'].cuda()
        utility = batch['utility'].float().cuda()
        secret  = batch['secret'].float().cuda()
        secret  = secret.view(secret.size(0))
        utility = utility.view(utility.size(0))

        # Sample noise as filter input
        batch_size = imgs.shape[0]
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        filter_imgs = filter(imgs, z, secret)

        if opt.use_real_fake:
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))
            filter_imgs = generator(filter_imgs, z, gen_secret)

        secret_pred_fix  = secret_classifier(filter_imgs)
        utility_pred_fix = utility_classifier(filter_imgs)

        def accuracy(pred, true):
            u   = true.cpu().numpy().flatten()
            p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
            acc = np.sum(u == p)/len(u)

            return acc

        running_secret_acc  += accuracy(secret_pred_fix, secret)
        running_utility_acc += accuracy(utility_pred_fix, utility)

    secret_acc  = running_secret_acc / len(valid_dataloader)
    utility_acc = running_utility_acc / len(valid_dataloader)

    return secret_acc, utility_acc, imgs, filter_imgs

def train(i_epoch):
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
        optimizer_f.zero_grad()

        # sample noise as filter input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # filter a batch of images
        filter_imgs = filter(imgs, z, secret)
        pred_secret = discriminator(filter_imgs)

        # loss measures filters's ability to fool the discriminator under constrained distortion
        ones = Variable(FloatTensor(secret.shape).fill_(1.0), requires_grad=False)
        target = ones-secret.float()
        target = target.view(target.size(0), -1)
        f_adversary_loss = adversarial_loss(pred_secret, target)
        f_distortion_loss = distortion_loss(filter_imgs, imgs)

        f_loss = f_adversary_loss + opt.lambd * torch.pow(torch.relu(f_distortion_loss-opt.eps), 2)

        f_loss.backward()
        optimizer_f.step()

        # ------------------------
        # Train Generator (Real/Fake)
        # ------------------------
        if opt.use_real_fake:
            optimizer_g.zero_grad()
            # sample noise as filter input
            z1 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

            # filter a batch of images
            filter_imgs = filter(imgs, z1, secret)

            # sample noise as generator input
            z2 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

            # sample secret
            gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))

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

        optimizer_d.zero_grad()

        pred_secret = discriminator(filter_imgs.detach())
        d_loss = adversarial_loss(pred_secret, secret)

        d_loss.backward()
        optimizer_d.step()

        # --------------------------------
        #  Train Discriminator (Real/Fake)
        # --------------------------------

        if opt.use_real_fake:
            optimizer_d_rf.zero_grad()

            real_pred_secret = discriminator_rf(imgs)
            fake_pred_secret = discriminator_rf(gen_imgs.detach())

            fake_secret = Variable(LongTensor(fake_pred_secret.size(0)).fill_(2.0), requires_grad=False)
            d_rf_loss_real = adversarial_rf_loss(real_pred_secret, secret.long())
            d_rf_loss_fake = adversarial_rf_loss(fake_pred_secret, fake_secret)

            d_rf_loss = (d_rf_loss_real + d_rf_loss_fake) / 2

            d_rf_loss.backward()
            optimizer_d_rf.step()

        if i_batch % opt.log_interval:
            writer.add_scalar('loss/d_loss', d_loss.item(), i_batch + i_epoch*len(train_dataloader))
            writer.add_scalar('loss/f_loss', f_loss.item(), i_batch + i_epoch*len(train_dataloader))
            if opt.use_real_fake:
                writer.add_scalar('loss/d_rf_loss', d_rf_loss.item(), i_batch + i_epoch*len(train_dataloader))
                writer.add_scalar('loss/g_loss', g_loss.item(), i_batch + i_epoch*len(train_dataloader))

#def save_images(imgs, filter_imgs, i_epoch):
#    """Saves a grid of generated digits ranging from 0 to n_classes"""
#    nb_samples = 8
#
#    imgs = imgs[:nb_samples]
#    filter_imgs = filter_imgs[:nb_samples]
#    diff_img = imgs-filter_imgs
#    sample_images = torch.cat((imgs, filter_imgs, diff_img))
#    save_dir = os.path.join(artifacts_path, "images")
#    os.makedirs(save_dir, exist_ok=True)
#    save_file = os.path.join(save_dir, "{}.png".format(i_epoch))
#
#    save_image(sample_images.data, save_file, nrow=nb_samples, normalize=True)

# ----------
#  Training
# ----------

for i_epoch in range(opt.n_epochs):
    # validate models
    secret_acc, utility_acc, imgs, filter_imgs = validate()
    print("secret_acc: ", secret_acc)
    print("utility_acc: ", utility_acc)

    # log results
    writer.add_scalar('valid/secret_acc', secret_acc, i_epoch)
    writer.add_scalar('valid/utility_acc', utility_acc, i_epoch)
    utils.save_images(imgs, filter_imgs, artifacts_path, i_epoch)

    # train models
    train(i_epoch)

    # save models
    utils.save_model(filter, os.path.join(artifacts_path, "filter.hdf5"))
    utils.save_model(discriminator, os.path.join(artifacts_path, "discriminator.hdf5"))
    utils.save_model(generator, os.path.join(artifacts_path, "generator.hdf5"))
    utils.save_model(discriminator_rf, os.path.join(artifacts_path, "discriminator_rf.hdf5"))

