import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, out_dim=1, activation='sigmoid'):
        super(Discriminator, self).__init__()
        self.activation = activation

        self.model = nn.Sequential(
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
        x = self.model(img.view(img.size(0), -1))
        if self.activation == 'sigmoid':
            x = nn.functional.sigmoid(x)
        return x


