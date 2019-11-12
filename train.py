import json
import tqdm
import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import utils

from torch.autograd import Variable
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class Trainer():
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.cuda = True if torch.cuda.is_available() else False
        self.opt = opt

        self.artifacts_path = 'artifacts/{}_eps_{}_lambd_{}_encoder_dim_{}/'.format(opt.name, opt.eps, opt.lambd, opt.encoder_dim)
        os.makedirs(self.artifacts_path, exist_ok=True)
        self.writer = SummaryWriter(self.artifacts_path)

        # models
        self.g1 = utils.get_generator_1(
            name          = 'fc_g1',
            input_shape   = (opt.channels, opt.img_size, opt.img_size),
            nb_classes    = opt.n_classes,
            encoder_dim   = opt.encoder_dim,
            embedding_dim = opt.embedding_dim,
            latent_dim    = opt.latent_dim
        )
        self.g2 = utils.get_generator_2(
            name          = 'fc_g2',
            input_shape   = (opt.channels, opt.img_size, opt.img_size),
            nb_classes    = opt.n_classes,
            encoder_dim   = opt.encoder_dim,
            embedding_dim = opt.embedding_dim,
            latent_dim    = opt.latent_dim
        )
        self.d1 = utils.get_discriminator_1(
            name        = 'fc_d1',
            input_shape = (opt.channels, opt.img_size, opt.img_size),
        )
        self.d2 = utils.get_discriminator_2(
            name          = 'fc_d2',
            input_shape   = (opt.channels, opt.img_size, opt.img_size),
            nb_classes    = opt.n_classes,
            embedding_dim = opt.embedding_dim,
            out_dim       = 3,
            activation    = None
        )

        # loss functions
        self.adversarial_loss = torch.nn.BCELoss()
        self.adversarial_rf_loss = torch.nn.CrossEntropyLoss()
        self.distortion_loss = torch.nn.MSELoss()

        if self.cuda:
            self.g1.cuda()
            self.g2.cuda()
            self.d1.cuda()
            self.d2.cuda()

            self.adversarial_loss.cuda()
            self.adversarial_rf_loss.cuda()
            self.distortion_loss.cuda()

        # optimizers
        if opt.train_together:
            self.optimizer_g = torch.optim.Adam(
                    list(self.g1.parameters()) + list(self.g2.parameters()),
                    lr=opt.lr, betas=(opt.b1, opt.b2))
        else:
            self.optimizer_g1 = torch.optim.Adam(
                self.g1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
            self.optimizer_g2 = torch.optim.Adam(
                self.g2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        self.optimizer_d1 = torch.optim.Adam(self.d1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_d2 = torch.optim.Adam(self.d2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        self.train_dataloader = utils.get_dataloader(
            'train', img_size=opt.img_size, in_memory=True, batch_size=opt.batch_size, shuffle=True)
        self.valid_dataloader = utils.get_dataloader(
            'valid', img_size=opt.img_size, in_memory=True, batch_size=opt.batch_size, shuffle=False)
        self.test_dataloader = utils.get_dataloader(
            'test', img_size=opt.img_size, in_memory=True, batch_size=opt.batch_size, shuffle=False)

        # TODO: make configurable in opt
        # fixed classifiers
        self.secret_classifier = utils.get_fix_discriminator(
            name = 'resnet_small',
            num_classes=2,
            pretrained=False,
            device='cuda:0',
            weights_path='./artifacts/fixed_resnet_small/classifier_secret_{}x{}.h5'.format(opt.img_size, opt.img_size)
        )
        self.utility_classifier = utils.get_fix_discriminator(
            name = 'resnet_small',
            num_classes=2,
            pretrained=False,
            device='cuda:0',
            weights_path='./artifacts/fixed_resnet_small/classifier_utility_{}x{}.h5'.format(opt.img_size, opt.img_size)
        )

    def load_weights(self):
        return

    def summary(self):
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("-----------------------")
        print("- Number of parameters:")
        print("-----------------------")
        print("g1: ", count_parameters(self.g1))
        print("g2: ", count_parameters(self.g2))
        print("d1: ", count_parameters(self.d1))
        print("d2: ", count_parameters(self.d2))

    def validate_epoch(self, i_epoch, verbose=False, save_images=False):
        running_secret_acc = 0.0
        running_utility_acc = 0.0

        for i_batch, batch in tqdm.tqdm(enumerate(self.valid_dataloader, 0)):
            imgs  = batch['image']
            utility = batch['utility'].float()
            secret  = batch['secret'].float()

            if self.cuda:
                imgs = imgs.cuda()
                utility = utility.cuda()
                secret = secret.cuda()

            secret  = secret.view(secret.size(0))
            utility = utility.view(utility.size(0))

            # Sample noise as filter input
            batch_size = imgs.shape[0]
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.opt.latent_dim))))

            filter_imgs = self.g1(imgs, z, secret)

            if self.opt.use_real_fake:
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.opt.latent_dim))))
                gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))
                filter_imgs = self.g2(filter_imgs, z, gen_secret)

            secret_pred_fix  = self.secret_classifier(filter_imgs)
            utility_pred_fix = self.utility_classifier(filter_imgs)

            def accuracy(pred, true):
                u   = true.cpu().numpy().flatten()
                p   = np.argmax(pred.cpu().detach().numpy(), axis=1)
                acc = np.sum(u == p)/len(u)

                return acc

            running_secret_acc  += accuracy(secret_pred_fix, secret)
            running_utility_acc += accuracy(utility_pred_fix, utility)

        secret_acc  = running_secret_acc / len(self.valid_dataloader)
        utility_acc = running_utility_acc / len(self.valid_dataloader)

        self.writer.add_scalar('acc/secret_acc', secret_acc, i_epoch)
        self.writer.add_scalar('acc/utility_acc', utility_acc, i_epoch)

        if verbose:
            print("secret acc: ", secret_acc)
            print("utility acc: ", utility_acc)

        save_dir = os.path.join(self.artifacts_path, 'images')
        if save_images:
            utils.save_images(imgs, filter_imgs, save_dir, i_epoch)

        return

    def train_epoch(self, i_epoch):
        for i_batch, batch in tqdm.tqdm(enumerate(self.train_dataloader)):
            imgs   = batch['image']
            secret = batch['secret'].float()
            secret = secret.view(secret.size(0))

            if self.cuda:
                imgs = imgs.cuda()
                secret = secret.cuda()

            batch_size = imgs.shape[0]

            # -----------------
            #  Train generator 1
            # -----------------
            if not self.opt.train_together:
                self.optimizer_g1.zero_grad()
            else:
                self.optimizer_g.zero_grad()

            # sample noise as filter input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.opt.latent_dim))))

            # filter a batch of images
            filter_imgs = self.g1(imgs, z, secret)
            pred_secret = self.d1(filter_imgs)

            # loss measures filters's ability to fool the discriminator under constrained distortion
            ones = Variable(FloatTensor(secret.shape).fill_(1.0), requires_grad=False)
            target = ones-secret.float()
            target = target.view(target.size(0), -1)
            f_adversary_loss = self.adversarial_loss(pred_secret, target)
            f_distortion_loss = self.distortion_loss(filter_imgs, imgs)

            g1_loss = f_adversary_loss + self.opt.lambd * torch.pow(torch.relu(f_distortion_loss-self.opt.eps), 2)

            if not self.opt.train_together:
                g1_loss.backward()
                self.optimizer_g1.step()

            # ------------------------
            # Train Generator (Real/Fake)
            # ------------------------
            if self.opt.use_real_fake:
                if not self.opt.train_together:
                    self.optimizer_g2.zero_grad()
                # sample noise as filter input
                z1 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.opt.latent_dim))))

                # filter a batch of images
                filter_imgs = self.g1(imgs, z1, secret)

                # sample noise as generator input
                z2 = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.opt.latent_dim))))

                # sample secret
                gen_secret = Variable(LongTensor(np.random.choice([0.0, 1.0], batch_size)))

                # generate a batch of images
                gen_imgs = self.g2(filter_imgs, z2, gen_secret)

                # loss measures generator's ability to fool the discriminator
                pred_secret = self.d2(gen_imgs, gen_secret)
                g_adversary_loss = self.adversarial_rf_loss(pred_secret, gen_secret)
                g_distortion_loss = self.distortion_loss(gen_imgs, imgs)

                g2_loss = g_adversary_loss + self.opt.lambd * torch.pow(torch.relu(g_distortion_loss-self.opt.eps), 2)

                if not self.opt.train_together:
                    g2_loss.backward()
                    self.optimizer_g2.step()

            if self.opt.train_together:
                g_loss = g1_loss + g2_loss
                g_loss.backward()
                self.optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_d1.zero_grad()

            pred_secret = self.d1(filter_imgs.detach())
            d_loss = self.adversarial_loss(pred_secret, secret)

            d_loss.backward()
            self.optimizer_d1.step()

            # --------------------------------
            #  Train Discriminator (Real/Fake)
            # --------------------------------

            if self.opt.use_real_fake:
                self.optimizer_d2.zero_grad()

                c_imgs = torch.cat((imgs, gen_imgs.detach()), axis=1)
                real_pred_secret = self.d2(imgs, secret.long())
                fake_pred_secret = self.d2(gen_imgs.detach(), gen_secret)

                fake_secret = Variable(LongTensor(fake_pred_secret.size(0)).fill_(2.0), requires_grad=False)
                d_rf_loss_real = self.adversarial_rf_loss(real_pred_secret, secret.long())
                d_rf_loss_fake = self.adversarial_rf_loss(fake_pred_secret, fake_secret)

                d_rf_loss = (d_rf_loss_real + d_rf_loss_fake) / 2

                d_rf_loss.backward()
                self.optimizer_d2.step()

            if i_batch % self.opt.log_interval:
                self.writer.add_scalar('loss/d_loss', d_loss.item(), i_batch + i_epoch*len(self.train_dataloader))
                if self.opt.train_together:
                    self.writer.add_scalar('loss/g_loss', g_loss.item(), i_batch + i_epoch*len(self.train_dataloader))
                else:
                    self.writer.add_scalar('loss/g1_loss', g1_loss.item(), i_batch + i_epoch*len(self.train_dataloader))
                    if self.opt.use_real_fake:
                        self.writer.add_scalar('loss/d_rf_loss', d_rf_loss.item(), i_batch + i_epoch*len(self.train_dataloader))
                        self.writer.add_scalar('loss/g2_loss', g2_loss.item(), i_batch + i_epoch*len(self.train_dataloader))


        return

    def save_train_state(self, i_epoch):
        # save opt
        with open(os.path.join(self.artifacts_path, 'hparams.json'), 'w') as f:
            json.dump(self.opt.__dict__, f)

        # save models
        utils.save_model(self.d1, os.path.join(self.artifacts_path, "d1.hdf5"))
        utils.save_model(self.d2, os.path.join(self.artifacts_path, "d2.hdf5"))
        utils.save_model(self.g1, os.path.join(self.artifacts_path, "g1.hdf5"))
        utils.save_model(self.g2, os.path.join(self.artifacts_path, "g2.hdf5"))

        # save epoch
        train_state = {
            'epoch' : i_epoch
        }
        with open(os.path.join(self.artifacts_path, 'train_state.json'), 'w') as f:
            json.dump(train_state, f)

        return

    def restore(self):
        return

    def predict(self):
        return
