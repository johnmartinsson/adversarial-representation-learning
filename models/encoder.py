import numpy as np
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, opt):
        super(ConvEncoder, self).__init__()

        def block(in_chs, out_chs):
            return [
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, (3, 3), padding=1),
                nn.LeakyReLU(0.2)
            ]

        self.flat_representation = nn.Sequential(
            *block(opt.channels, 256),
            *block(256, 256),
            #*block(256, 256),
            nn.MaxPool2d(2, 2),
            *block(256, 128),
            *block(128, 128),
            #*block(128, 128),
            nn.MaxPool2d(2, 2),
            *block(128, 64),
            *block(64, 16),
            #*block(64, 64),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.model = nn.Sequential(
            nn.Linear((opt.img_size//8)**2 * 16, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, opt.encoder_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, img): #, labels):
        img_emb = self.flat_representation(img)

        # Concatenate label embedding and image to produce input
        return self.model(img_emb)

class Encoder(nn.Module):
    def __init__(self, input_shape, encoder_dim):
        super(Encoder, self).__init__()

        def block(in_feat, out_feat, dropout_rate):
            return [
                nn.Linear(in_feat, out_feat),
                nn.Dropout(dropout_rate),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        img_shape = input_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            *block(512, 512, 0.1),
            *block(512, 512, 0.1),
            *block(512, 256, 0.1),
            *block(256, encoder_dim, 0.1),
        )

    def forward(self, img): #, labels):
        # Concatenate label embedding and image to produce input
        return self.model(img)


