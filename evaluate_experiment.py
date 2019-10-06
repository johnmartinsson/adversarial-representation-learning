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

class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

def evaluate(hparams):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hparams.device = device
    experiment_path = os.path.join('artifacts', hparams.experiment_name)
    print("evaluating : ", experiment_path)

    print("loading models ...")
    G1 = utils.get_generator_model(
        name=hparams.generator_name,
        device = hparams.device,
        noise_dim = hparams.noise_dim,
        additive_noise = hparams.use_additive_noise,
        weights_path = os.path.join(experiment_path, 'G1.hdf5'))
    G2 = utils.get_generator_model(
        name=hparams.generator_name,
        device = hparams.device,
        noise_dim = hparams.noise_dim,
        additive_noise = hparams.use_additive_noise,
        weights_path = os.path.join(experiment_path, 'G2.hdf5'))

    Ds = utils.get_discriminator_model(
        name=hparams.discriminator_name,
        num_classes=2,
        pretrained=False,
        device=hparams.device,
        weights_path = os.path.join(experiment_path, 'Ds.hdf5'))

    if hparams.use_weighted_squared_error:
        G_w = utils.get_generator_model(
            name='unet_weight',
            device = hparams.device,
            noise_dim = None,
            additive_noise = hparams.use_additive_noise,
            weights_path = os.path.join(experiment_path, 'G_w.hdf5'))
    else:
        G_w = None

    Ds_fix = utils.get_discriminator_model('resnet_small', num_classes=2,
            pretrained=False, device=hparams.device,
            weights_path='./artifacts/fixed_resnet_small/classifier_secret_64x64.h5')
    Du_fix = utils.get_discriminator_model('resnet_small', num_classes=2,
            pretrained=False, device=hparams.device,
            weights_path='./artifacts/fixed_resnet_small/classifier_utility_64x64.h5')

    print("loading dataset ...")
    celeba_testdataset = celeba.CelebADataset(
        split='test',
        in_memory=True,
        input_shape=((hparams.image_width, hparams.image_height)),
        utility_attr=hparams.utility_attr,
        secret_attr=hparams.secret_attr,
        transform=celeba.ToTensor())
    testloader = DataLoader(celeba_testdataset, batch_size=32,
            num_workers=8)

    print("running evaluation ...")
    (secret_acc, utility_acc), val_result = validate(G1, G2, G_w, Ds_fix, Du_fix, testloader, hparams)
    print(secret_acc, utility_acc)

    # save result
    print("saving results ...")
    with open(os.path.join(experiment_path, 'evaluation_results.json'), 'w') as f:
        json.dump({'val_secret_acc':secret_acc, 'val_utility_acc':utility_acc}, f, indent=2)

    names = ['real_images', 'representations', 'fake_images']
    for i, name in enumerate(names):
        grid = torchvision.utils.make_grid(val_result[i])
        grid = grid.detach().cpu().numpy().transpose(1,2,0) # CxWxH -> WxHxC
        plt.title("alpha = {}, g_lr = {}, ds_lr = {}".format(
            hparams.alpha, hparams.g_lr, hparams.ds_lr))
        plt.imshow(grid)
        plt.savefig(os.path.join(experiment_path, '{}.png'.format(name)))

    if hparams.use_weighted_squared_error:
        grid = torchvision.utils.make_grid(val_result[3])
        grid = grid.detach().cpu().numpy().transpose(1,2,0) # CxWxH -> WxHxC
        plt.imshow(grid)
        plt.title("alpha = {}, g_lr = {}, ds_lr = {}".format(
            hparams.alpha, hparams.g_lr, hparams.ds_lr))
        plt.savefig(os.path.join(experiment_path, '{}.png'.format('weights')))

def main():
    parser = ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.experiment_dir, 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    print(hparams)
    evaluate(Map(hparams))

if __name__ == '__main__':
    main()
