import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, padding=1),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(),
        nn.Conv2d(channels_out, channels_out, 3, padding=1),
        nn.BatchNorm2d(channels_out),
        nn.ReLU()
    )

class UNet2(nn.Module):
    def __init__(self, channels_in, channels_out, chs=[8, 16, 32, 64, 128], image_width=64, image_height=64, noise_dim=10, additive_noise=True, device='cpu'):
        super().__init__()

        self.width  = image_width
        self.height = image_height
        self.additive_noise = additive_noise

        # noise projection layer
        if noise_dim is not None:
            if not additive_noise:
                self.project_noise = nn.Linear(noise_dim, image_width*image_height)
                down_conv_first = double_conv(channels_in+1, chs[0])
            else:
                self.project_noise = nn.Linear(noise_dim, channels_in*image_width*image_height)
                down_conv_first = double_conv(channels_in, chs[0])
        else:
            down_conv_first = double_conv(channels_in, chs[0])

        self.pool_down1  = nn.MaxPool2d(2, stride=2)

        assert len(chs) > 1


        down_convs = [double_conv(chs[i], chs[i+1]).to(device) for i in range(0, len(chs)-1)]
        up_convs   = [double_conv(chs[i+1]+chs[i], chs[i+1]).to(device) for i in range(0, len(chs)-1)]

        self.down_conv_first = down_conv_first
        self.up_conv_last    = nn.Conv2d(chs[1], channels_out, kernel_size=1)
        self.down_convs      = down_convs
        self.up_convs        = up_convs
        self.pool            = nn.MaxPool2d(2, stride=2)

    def forward(self, x, z=None):

        if z is not None:
            if self.additive_noise:
                noise = self.project_noise(z).reshape(x.shape)
                x = x + noise
            else:
                noise = self.project_noise(z).reshape(x.shape[0], 1, x.shape[2], x.shape[3])
                x = torch.cat((x, noise), dim=1) # concatenate along channel dimension

        # top down
        conv_downs = [self.down_conv_first(x)] # 1, 2, 4, 8, ...
        pools      = [] # 2, 4, 8, ...
        for i_conv in range(len(self.down_convs)):
            pool      = self.pool(conv_downs[i_conv])
            conv_down = self.down_convs[i_conv](pool)
            conv_downs.append(conv_down)

        # bottom up
        for i_conv in range(len(conv_downs))[::-1][:-1]:
            conv_up = F.interpolate(conv_downs[i_conv], scale_factor=2, mode='nearest')
            conv_up = torch.cat((conv_downs[i_conv-1], conv_up), dim=1)
            conv_up = self.up_convs[i_conv-1](conv_up)

        conv_up = self.up_conv_last(conv_up)
        #x = torch.sigmoid(conv_up)
        x = torch.tanh(conv_up)

        return x

class UNet(nn.Module):
    def __init__(self, channels_in, channels_out, chs=[8, 16, 32, 64, 128], image_width=64, image_height=64, noise_dim=10, activation='tanh', additive_noise=True):
        super().__init__()

        self.width  = image_width
        self.height = image_height
        self.additive_noise = additive_noise
        self.activation = activation

        # noise projection layer
        if noise_dim is not None:
            if not additive_noise:
                self.project_noise = nn.Linear(noise_dim, image_width*image_height)
                self.dconv_down1 = double_conv(channels_in+1, chs[0])
            else:
                self.project_noise = nn.Linear(noise_dim, channels_in*image_width*image_height)
                self.dconv_down1 = double_conv(channels_in, chs[0])
        else:
            self.dconv_down1 = double_conv(channels_in, chs[0])

        self.pool_down1  = nn.MaxPool2d(2, stride=2)

        self.dconv_down2 = double_conv(chs[0], chs[1])
        self.pool_down2  = nn.MaxPool2d(2, stride=2)

        self.dconv_down3 = double_conv(chs[1], chs[2])
        self.pool_down3  = nn.MaxPool2d(2, stride=2)

        self.dconv_down4 = double_conv(chs[2], chs[3])
        self.pool_down4  = nn.MaxPool2d(2, stride=2)

        self.dconv_down5 = double_conv(chs[3], chs[4])

        self.dconv_up5 = double_conv(chs[4]+chs[3], chs[4])
        self.dconv_up4 = double_conv(chs[3]+chs[2], chs[3])
        self.dconv_up3 = double_conv(chs[2]+chs[1], chs[2])
        self.dconv_up2 = double_conv(chs[1]+chs[0], chs[1])
        self.dconv_up1 = nn.Conv2d(chs[1], channels_out, kernel_size=1)

    def forward(self, x, z=None):

        if z is not None:
            if self.additive_noise:
                noise = self.project_noise(z).reshape(x.shape)
                x = x + noise
            else:
                noise = self.project_noise(z).reshape(x.shape[0], 1, x.shape[2], x.shape[3])
                x = torch.cat((x, noise), dim=1) # concatenate along channel dimension

        conv1_down = self.dconv_down1(x)
        pool1 = self.pool_down1(conv1_down)

        conv2_down = self.dconv_down2(pool1)
        pool2 = self.pool_down2(conv2_down)

        conv3_down = self.dconv_down3(pool2)
        pool3 = self.pool_down3(conv3_down)

        conv4_down = self.dconv_down4(pool3)
        pool4 = self.pool_down4(conv4_down)

        conv5_down = self.dconv_down5(pool4)

        conv5_up = F.interpolate(conv5_down, scale_factor=2, mode='nearest')
        conv5_up = torch.cat((conv4_down, conv5_up), dim=1)
        conv5_up = self.dconv_up5(conv5_up)

        conv4_up = F.interpolate(conv4_down, scale_factor=2, mode='nearest')
        conv4_up = torch.cat((conv3_down, conv4_up), dim=1)
        conv4_up = self.dconv_up4(conv4_up)

        conv3_up = F.interpolate(conv3_down, scale_factor=2, mode='nearest')
        conv3_up = torch.cat((conv2_down, conv3_up), dim=1)
        conv3_up = self.dconv_up3(conv3_up)

        conv2_up = F.interpolate(conv2_down, scale_factor=2, mode='nearest')
        conv2_up = torch.cat((conv1_down, conv2_up), dim=1)
        conv2_up = self.dconv_up2(conv2_up)

        conv1_up = self.dconv_up1(conv2_up)
        if self.activation == 'sigmoid':
            x = torch.sigmoid(conv1_up)
        else:
            x = torch.tanh(conv1_up)

        return x
