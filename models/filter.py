import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder

class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.embedding_dim)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, img, noise, label):
        # Concatenate label embedding and image to produce input
        img = img.view(img.size(0), -1)
        encoding = self.encoder(img)
        encoding = torch.cat((encoding, self.label_emb(label.long()), noise), -1)
        diff_img = self.decoder(encoding)
        img = torch.nn.functional.sigmoid(img + diff_img)
        img = img.view(img.size(0), *img_shape)
        return img


