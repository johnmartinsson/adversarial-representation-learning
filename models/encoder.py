import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        def block(in_feat, out_feat, dropout_rate):
            return [
                nn.Linear(in_feat, out_feat),
                nn.Dropout(dropout_rate),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            *block(512, 512, 0.1),
            *block(512, 512, 0.1),
            *block(512, 256, 0.1),
            *block(256, opt.encoder_dim, 0.1),
        )

    def forward(self, img): #, labels):
        # Concatenate label embedding and image to produce input
        return self.model(img)


