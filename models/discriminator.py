import torch
import numpy as np
import torch.nn as nn

class ConvDiscriminator(nn.Module):
    def __init__(self, opt, out_dim=1, activation='sigmoid'):
        super(ConvDiscriminator, self).__init__()
        self.activation = activation

        self.label_embedding = nn.Embedding(opt.n_classes, opt.embedding_dim)

        def block(in_chs, out_chs):
            return (
                #nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, (3, 3), padding=1),
                nn.LeakyReLU(0.2)
            )

        self.flat_representation = nn.Sequential(
            *block(opt.channels, 256),
            *block(256, 256),
            *block(256, 256),
            nn.MaxPool2d(2, 2),
            *block(256, 128),
            *block(128, 128),
            *block(128, 128),
            nn.MaxPool2d(2, 2),
            *block(128, 64),
            *block(64, 64),
            *block(64, 64),
            nn.MaxPool2d(2, 2),
            *block(64, 16),
            *block(16, 16),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.predict = nn.Sequential(
            nn.Linear(opt.embedding_dim + (opt.img_size//16)**2 * 16, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, img, label):
        label_emb = self.label_embedding(label)
        img_emb   = self.flat_representation(img)
        x         = self.predict(torch.cat((img_emb, label_emb), -1))
        if self.activation == 'sigmoid':
            x = nn.functional.sigmoid(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_shape, out_dim=1, activation='sigmoid'):
        super(Discriminator, self).__init__()
        self.activation = activation

        img_shape = input_shape

        self.predict = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, img):
        d_in = img.view(img.size(0), -1)
        x = self.predict(d_in)
        if self.activation == 'sigmoid':
            x = nn.functional.sigmoid(x)
        return x

class SecretDiscriminator(nn.Module):
    def __init__(self, input_shape, out_dim=1, activation='sigmoid'):
        super(SecretDiscriminator, self).__init__()
        self.activation = activation

        img_shape = input_shape # (opt.channels, opt.img_size, opt.img_size)

        self.predict = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, img):
        d_in = img.view(img.size(0), -1)
        x = self.predict(d_in)
        if self.activation == 'sigmoid':
            x = nn.functional.sigmoid(x)
        elif self.activation == 'softmax':
            x = nn.functional.softmax(x)
        else:
            return x

class ConditionalDiscriminator(nn.Module):
    def __init__(self, opt, out_dim=1, activation='sigmoid'):
        super(ConditionalDiscriminator, self).__init__()
        self.activation = activation

        img_shape = (opt.channels, opt.img_size, opt.img_size)
        self.label_embedding = nn.Embedding(opt.n_classes, opt.embedding_dim)

        self.model = nn.Sequential(
            nn.Linear(opt.embedding_dim + 2*int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, img, img_c, label):
        label_emb = self.label_embedding(label)
        img = img.view(img.size(0), -1)
        img_c = img_c.view(img.size(0), -1)
        d_in = torch.cat((label_emb, img, img_c), -1)
        x = self.model(d_in)
        if self.activation == 'sigmoid':
            x = nn.functional.sigmoid(x)
        return x
