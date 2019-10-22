import numpy as np
import torch.nn as nn

class ConvDecoder(nn.Module):
    def __init__(self, opt):
        super(ConvDecoder, self).__init__()

        def block(in_chs, out_chs):
            return [
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, (3, 3), padding=1),
                nn.LeakyReLU(0.2),
            ]

        self.img_size = opt.img_size

        self.decode_1 = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.encoder_dim + opt.embedding_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, (opt.img_size//8)**2 * 16),
            nn.LeakyReLU(0.2),
        )

        self.decode_2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            *block(16, 64),
            *block(64, 128),
            nn.Upsample(scale_factor=2),
            *block(128, 128),
            *block(128, 256),
            nn.Upsample(scale_factor=2),
            *block(256, 256),
            *block(256, opt.channels),
        )

    def forward(self, encoding):
        x = self.decode_1(encoding)
        x = x.view((x.size(0), 16, self.img_size//8, self.img_size//8))
        x = self.decode_2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        img_shape = (opt.channels, opt.img_size, opt.img_size)
        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.encoder_dim + opt.embedding_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
        )

    def forward(self, encoding):
        return self.model(encoding)


