import os
from argparse import ArgumentParser
import json
import models.utils as utils
import torch
from torch.utils.data import DataLoader
import datasets.celeba as celeba
from adversarial_bottleneck_experiment import validate
import torchvision

import matplotlib.pyplot as plt

def evaluate(hparams):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hparams.device = device
    experiment_path = os.path.join('artifacts', hparams.experiment_name)
    G1 = utils.get_generator_model(
        name=hparmas.generator_name,
        device = hparams.device,
        noise_dim = hparams.noise_dim,
        additive_noise = hparams.use_additive_noise,
        weights_path = os.path.join(experiment_path, 'G1.h5'))
    G2 = utils.get_generator_model(
        name=hparmas.generator_name,
        device = hparams.device,
        noise_dim = hparams.noise_dim,
        additive_noise = hparams.use_additive_noise,
        weights_path = os.path.join(experiment_path, 'G2.h5'))

    Ds = utils.get_discriminator_model(
        name=hparams.discriminator_name,
        num_classes=2,
        pretrained=False,
        device=hparams.device,
        weights_path = os.path.join(experiment_path, 'Ds.h5'))

    if hparams.use_weighted_squared_error:
        G_w = utils.get_generator_model(
            name='unet_weight',
            device = hparams.device,
            noise_dim = None,
            additive_noise = hparams.use_additive_noise,
            weights_path = os.path.join(experiment_path, 'G_w.h5'))
    else:
        G_w = None

    Ds_fix = utils.get_discriminator_model('resnet_small', num_classes=2,
            pretrained=False, device=hparams.device,
            weights_path='./artifacts/fixed_resnet_small/classifier_secret_64x64.h5')
    Du_fix = utils.get_discriminator_model('resnet_small', num_classes=2,
            pretrained=False, device=hparams.device,
            weights_path='./artifacts/fixed_resnet_small/classifier_utility_64x64.h5')

    celeba_testdataset = celeba.CelebADataset(
        split='test',
        in_memory=True,
        input_shape=((hparams.image_width, hparams.image_height)),
        utility_attr=hparams.utility_attr,
        secret_attr=hparams.secret_attr,
        transform=celeba.ToTensor())
    testloader = DataLoader(celeba_testdataset, batch_size=32,
            num_workers=8)

    (secret_acc, utility_acc), val_result = validate(G1, G2, G_w, Ds_fix, Du_fix, testloader, hparams)
    print(secret_acc, utility_acc)

    # save result
    with open(os.path.join(experiment_path, 'evaluation_results.json'), 'w') as f:
        json.dump({'val_secret_acc':secret_acc, 'val_utility_acc':utility_acc})

    names = ['real_images', 'representations', 'fake_images']
    for i, name in enumerate(names):
        grid = torchvision.utils.make_grid(val_result[i])
        grid = grid.cpu().numpy()
        plt.imshow(grid)
        plt.savefig(os.path.join(experiment_path, '{}.png'.format(name)))

    if hparams.use_weighted_squared_error:
        grid = torchvision.utils.make_grid(val_result[3])
        grid = grid.cpu().numpy()
        plt.imshow(grid)
        plt.savefig(os.path.join(experiment_path, '{}.png'.format(name)))

def main():
    parser = ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    hparams = parser.parse_args()

    evaluate(hparams)

if __name__ == '__main__':
    main()
