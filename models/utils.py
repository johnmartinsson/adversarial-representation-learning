import torch
import torch.nn as nn
import torchvision.models as models
from models.unet import UNet

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


