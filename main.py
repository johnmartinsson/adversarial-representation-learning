import argparse
from train import Trainer

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--eps", type=float, default=0.001, help="distortion budget")
parser.add_argument("--lambd", type=float, default=100000.0, help="squared penalty")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--encoder_dim", type=int, default=256, help="dimensionality of the representation space")
parser.add_argument("--embedding_dim", type=int, default=16, help="dimensionality of embedding space")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--log_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--name", type=str, default='default', help="experiment name")
parser.add_argument("--use_real_fake", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--train_together", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("--mode", type=str, default='train')
opt = parser.parse_args()
print(opt)

# ----------
#  Training
# ----------

trainer = Trainer(opt)

trainer.summary()

if opt.mode == 'train':
    for i_epoch in range(opt.n_epochs):
        # validate models
        trainer.validate_epoch(i_epoch, verbose=True, save_images=True)

        # train models
        trainer.train_epoch(i_epoch)

        # save models
        trainer.save_train_state(i_epoch)
else:
    print(opt.mode, " not defined.")
