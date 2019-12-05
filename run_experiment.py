import os
import time
import subprocess
from argparse import ArgumentParser
from collections import deque

def train_classifiers(mode):
    secret_attributes = [
        'Attractive'
        'Young',
        'Male',
        'Wearing_Lipstick',
        'Wavy_Hair',
        'Heavy_Makeup',
        'Pointy_Nose',
        'High_Cheekbones',
        'Mouth_Slightly_Open',
        'Rosy_Cheeks',
        'Oval_Face'
    ]

    stack = deque()

    for attr in secret_attributes:
        experiment_name = os.path.join('fixed_classifiers', attr)
        artifacts_dir = os.path.join('artifacts', experiment_name)
        args = [
            'train_classifiers.py',
            '--classifier_name', 'resnet18',
            '--image_width', '64',
            '--image_height', '64',
            '--max_epochs', '10',
            '--batch_size', '256',
            '--experiment_name', experiment_name,
            '--attr', attr,
        ]
        stack.append((args, artifacts_dir))
    return stack

def secret_attributes():
    secret_attributes = [
        'Attractive'
        'Young',
        'Male',
        'Wearing_Lipstick',
        'Wavy_Hair',
        'Heavy_Makeup',
        'Pointy_Nose',
        'High_Cheekbones',
        'Mouth_Slightly_Open',
        'Rosy_Cheeks',
        'Oval_Face',
    ]
    stack = deque()

    for secret_attribute in secret_attributes:
        artifacts_dir = os.path.join('artifacts', 'secret_attributes', secret_attribute, '0')

        args = [
            'pcgan_v2.py',
            '--artifacts_dir', artifacts_dir,
            '--generator_name', 'unet_generator',
            '--discriminator_name', 'fc_discriminator',
            '--in_memory', 'false',
            '--secret_attr', secret_attribute,
            '--mode', mode,
        ]
        stack.append((args, artifacts_dir))

    return stack

def uneven_discriminator_and_generator_training(mode):
    discriminator_updates = [2, 4, 6, 8]
    stack = deque()

    for discriminator_update in discriminator_updates:
        for i in range(2): # run each experiment twice
            artifacts_dir = os.path.join('artifacts', 'uneven_discriminator_and_generator_training', 'discriminator_update_{}'.format(discriminator_update), str(i))

            args = [
                'pcgan_v2.py',
                '--artifacts_dir', artifacts_dir,
                '--discriminator_name', 'resnet_small_discriminator',
                '--generator_name', 'unet_generator',
                '--in_memory', 'true',
                '--mode', mode,
                '--discriminator_update_interval', str(discriminator_update)
            ]

            stack.append((args, artifacts_dir))
    return stack

def fc_vs_conv_discriminator_and_generator(mode):
    discriminator_names = ['fc_discriminator', 'resnet18_discriminator']
    generator_names     = ['fc_generator', 'unet_generator']
    stack = deque()

    for discriminator_name in discriminator_names:
        for generator_name in generator_names:
            artifacts_dir = os.path.join('artifacts', 'fc_vs_conv_discriminator_and_generator', discriminator_name + '_' + generator_name, '0')

            args = [
                'pcgan_v2.py',
                '--artifacts_dir', artifacts_dir,
                '--generator_name', generator_name,
                '--discriminator_name', discriminator_name,
                '--in_memory', 'false',
                '--mode', mode,
            ]
            stack.append((args, artifacts_dir))

    return stack

def spawn_experiment(args, gpu, artifacts_dir):
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    print("spawning experiment: {} ...".format(artifacts_dir))
    #print(args)
    my_env = os.environ.copy()
    my_env['CUDA_VISIBLE_DEVICES'] = gpu
    p = subprocess.Popen(['python'] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=my_env)

    return p

def main():
    parser = ArgumentParser()
    parser.add_argument('--gpus', nargs='+', help='gpus to run on', required=True)
    parser.add_argument('--experiment_name', type=str, help='name of experiment', required=True)
    parser.add_argument('--mode', type=str, default='train', help='train/evaluate')
    args = parser.parse_args()
    print(args)

    experiment_stack_builders = {
        'fc_vs_conv_discriminator_and_generator' : fc_vs_conv_discriminator_and_generator,
        'secret_attributes' : secret_attributes,
        'train_classifiers' : train_classifiers,
        'uneven_discriminator_and_generator_training' : uneven_discriminator_and_generator_training,
    }

    gpu_busy = {}

    gpus = args.gpus
    stack = experiment_stack_builders[args.experiment_name](args.mode)
    nb_experiments = len(stack)

    for gpu in gpus:
        (args, artifacts_dir) = stack.pop()
        p = spawn_experiment(args, gpu, artifacts_dir)
        gpu_busy[gpu] = p

    while len(stack) > 0:
        print("check if GPU available. {}/{}".format(nb_experiments-len(stack), nb_experiments))
        time.sleep(60)
        for gpu in gpus:
            p = gpu_busy[gpu]
            if not p.poll() is None:
                (args, artifacts_dir) = stack.pop()
                p = spawn_experiment(args, gpu, artifacts_dir)
                gpu_busy[gpu] = p

if __name__ == '__main__':
    main()
