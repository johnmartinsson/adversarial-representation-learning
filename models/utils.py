import os
import torch
import torch.nn as nn
import torchvision.models as models
from models.unet import UNet
from torchvision.utils import save_image

def save_images(imgs, filter_imgs, artifacts_path, i_epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    nb_samples = 8

    imgs = imgs[:nb_samples]
    filter_imgs = filter_imgs[:nb_samples]
    diff_img = torch.norm(imgs-filter_imgs, dim=1)
    #diff_img = torch.sum(diff_img, dim=1)
    diff_img = diff_img.view((diff_img.size(0), 1, diff_img.size(1), diff_img.size(2)))
    diff_img = torch.cat((diff_img, diff_img, diff_img), dim=1)
    sample_images = torch.cat((imgs, filter_imgs, diff_img))
    save_dir = os.path.join(artifacts_path, "images")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "{}.png".format(i_epoch))

    save_image(sample_images.data, save_file, nrow=nb_samples, normalize=True)

def save_images_2(imgs, filter_imgs, filter_imgs_0, filter_imgs_1, save_dir, i_batch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    nb_samples = 8

    imgs = imgs[:nb_samples]
    #filter_imgs = filter_imgs[:nb_samples]
    filter_imgs_0 = filter_imgs_0[:nb_samples]
    filter_imgs_1 = filter_imgs_1[:nb_samples]
    sample_images = torch.cat((imgs, filter_imgs_0, filter_imgs_1))
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "{}.png".format(i_batch))

    save_image(sample_images.data, save_file, nrow=nb_samples, normalize=True)


def save_model(model, path):
    torch.save(model.state_dict(), path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_generator_model(name, device, noise_dim, additive_noise, image_width=64, image_height=64, weights_path=None):
    if name == 'unet_weight':
        model = UNet(3, 1, [8, 16, 32, 64, 128], image_width, image_height, noise_dim, 'sigmoid', additive_noise)
    if name == 'unet_small':
        model = UNet(3, 3, [16, 32, 64, 128, 256], image_width, image_height, noise_dim, 'sigmoid', additive_noise)

    if not weights_path is None:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model.to(device)
    return model

def get_discriminator_model(name, num_classes, pretrained, device, weights_path=None):
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
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)

    # if weights path specified load model
    if not weights_path is None:
        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model


