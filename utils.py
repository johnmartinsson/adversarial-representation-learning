import os
import torch

from models.filter import Filter
from models.discriminator import SecretDiscriminator
from models.discriminator import Discriminator
import datasets.celeba as celeba

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image

def save_model(model, path):
    torch.save(model.state_dict(), path)

def save_images(imgs, filter_imgs, save_dir, i_epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    nb_samples = 8

    imgs = imgs[:nb_samples]
    filter_imgs = filter_imgs[:nb_samples]
    diff_img = torch.norm(imgs-filter_imgs, dim=1)
    diff_img = diff_img.view((diff_img.size(0), 1, diff_img.size(1), diff_img.size(2)))
    diff_img = torch.cat((diff_img, diff_img, diff_img), dim=1)
    sample_images = torch.cat((imgs, filter_imgs, diff_img))
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "{}.png".format(i_epoch))

    save_image(sample_images.data, save_file, nrow=nb_samples, normalize=True)

def get_discriminator_1(name, input_shape):
    if name == 'fc_d1':
        return SecretDiscriminator(input_shape)
    else:
        raise ValueError("Not supported model: ", name)

def get_discriminator_2(name, input_shape, nb_classes, embedding_dim, out_dim, activation):
    if name == 'fc_d2':
        return Discriminator(input_shape, nb_classes, embedding_dim, out_dim=out_dim, activation=activation)
    else:
        raise ValueError("Not supported model: ", name)

def get_generator_1(name, input_shape, nb_classes, encoder_dim, embedding_dim, latent_dim):
    if name == 'fc_g1':
        return Filter(input_shape, nb_classes, encoder_dim, embedding_dim, latent_dim)
    else:
        raise ValueError("Not supported model: ", name)

def get_generator_2(name, input_shape, nb_classes, encoder_dim, embedding_dim, latent_dim):
    if name == 'fc_g2':
        return Filter(input_shape, nb_classes, encoder_dim, embedding_dim, latent_dim)
    else:
        raise ValueError("Not supported model: ", name)

def get_fix_discriminator(name, num_classes, pretrained, device, weights_path=None):
    if name == 'resnet_small':
        assert(not pretrained)
        model = models.ResNet(models.resnet.BasicBlock, [1, 1, 1, 1])
    if name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    if name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    if name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    
    # replace last layer
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.to(device)

    # if weights path specified load model
    if not weights_path is None:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model

def get_dataloader(split, img_size, in_memory, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        celeba.CelebADataset(
            split=split,
            in_memory=in_memory,
            input_shape=(img_size, img_size),
            utility_attr='Male',
            secret_attr='Smiling',
            transform=transforms.Compose([
                celeba.ToTensor(),
        ])),
        batch_size=batch_size,
        shuffle=shuffle,
    )


