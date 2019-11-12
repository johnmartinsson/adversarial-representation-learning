import torch
import torch.nn as nn
from models.encoder import Encoder, ConvEncoder
from models.decoder import Decoder, ConvDecoder

class ConvFilter(nn.Module):
    def __init__(self, opt):
        super(ConvFilter, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.embedding_dim)
        self.encoder = ConvEncoder(opt)
        self.decoder = ConvDecoder(opt)
        self.blend = nn.Sequential(
            nn.Conv2d(opt.channels*2, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img, noise, label):
        # Concatenate label embedding and image to produce input
        encoding = self.encoder(img) # flat encoding
        encoding = torch.cat((encoding, self.label_emb(label.long()), noise), -1)
        diff_img = self.decoder(encoding) # image shape
        img      = self.blend(torch.cat((img, diff_img), dim=1))
        return img

class Filter(nn.Module):
    def __init__(self, input_shape, nb_classes, encoder_dim, embedding_dim, latent_dim):
        super(Filter, self).__init__()

        self.label_emb = nn.Embedding(nb_classes, embedding_dim)
        self.encoder = Encoder(input_shape, encoder_dim)
        self.decoder = Decoder(input_shape, encoder_dim, embedding_dim, latent_dim)
        self.img_shape = input_shape

    def forward(self, img, noise, label):
        # Concatenate label embedding and image to produce input
        img = img.view(img.size(0), -1)
        encoding = self.encoder(img)
        encoding = torch.cat((encoding, self.label_emb(label.long()), noise), -1)
        diff_img = self.decoder(encoding)
        img = torch.nn.functional.sigmoid(img + diff_img)
        img = img.view(img.size(0), *self.img_shape)
        return img


