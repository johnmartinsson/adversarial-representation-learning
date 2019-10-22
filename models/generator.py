import argparse
import torch
import numpy as np
from models.filter import Filter
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def default_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--encoder_dim", type=int, default=128, help="dimensionality of the representation space")
    parser.add_argument("--embedding_dim", type=int, default=256, help="dimensionality of embedding space")
    parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--evaluate", type=str2bool, nargs='?', const=True, default=False)
    opt = parser.parse_args()

    return opt

class Generator(torch.nn.Module):
    def __init__(self, opt, filter_1_path, filter_2_path):
        super(Generator, self).__init__()

        self.latent_dim = opt.latent_dim
        self.filter_1 = Filter(opt)
        self.filter_2 = Filter(opt)

        self.filter_1.load_state_dict(torch.load(filter_1_path))
        self.filter_2.load_state_dict(torch.load(filter_2_path))

    def generate(imgs, secrets, gen_secrets):

        z1 = Variable(FloatTensor(np.random.normal(0, 1, (imgs.size(0), self.latent_dim))))
        filter_imgs = self.filter_1(imgs, z1, secrets)
        z2 = Variable(FloatTensor(np.random.normal(0, 1, (imgs.size(0), self.latent_dim))))
        filter_imgs = self.filter_2(filter_imgs, z2, gen_secrets)

        return filter_imgs
