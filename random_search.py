import os
import time
import subprocess
from argparse import ArgumentParser
import numpy as np
from adversarial_bottleneck_experiment import run_experiment

def build_args_1(gpu, experiment_id):
    alpha = np.random.uniform(1, 20)
    ds_lr = np.random.choice([1e-2, 1e-3, 1e-4])
    g_lr = np.random.choice([1e-2, 1e-3, 1e-4])

    experiment_name = 'random_search_1/{}'.format(experiment_id)
    args = [
        'adversarial_bottleneck_experiment.py',
        '--alpha', str(alpha),
        '--ds_lr', str(ds_lr),
        '--g_lr', str(g_lr),
        '--device', gpu,
        '--experiment_name', experiment_name,
        '--max_epochs', str(100),
    ]

    return args, experiment_name

def build_args_2(gpu, experiment_id):
    alpha = np.random.uniform(1, 20)
    ds_lr = np.random.choice([1e-2, 1e-3, 1e-4])
    g_lr = np.random.choice([1e-2, 1e-3, 1e-4])
    weight_budget = np.random.choice([0.5, 0.10, 0.15, 0.20, 0.30])

    experiment_name = 'random_search_2/{}'.format(experiment_id)
    args = [
        'adversarial_bottleneck_experiment.py',
        '--alpha', str(alpha),
        '--ds_lr', str(ds_lr),
        '--g_lr', str(g_lr),
        '--device', gpu,
        '--experiment_name', experiment_name,
        '--max_epochs', str(100),
        '--use_weighted_squared_error', 'True',
        '--weight_budget', str(weight_budget),
    ]

    return args, experiment_name

def get_sampling_method(name):
    if name == 'build_args_1':
        return build_args_1
    elif name == 'build_args_2':
        return build_args_2
    else:
        raise ValueError("sampling method not supported ...")
    return sampling_method

def spawn_experiment(sampling_method, gpu, experiment_counter):
    args, experiment_name = sampling_method(gpu, experiment_counter)

    experiment_path = os.path.join('artifacts', experiment_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    print("spawning experiment: {} ...".format(experiment_name))
    print(args)
    p = subprocess.Popen(['python'] + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return p

def main():
    parser = ArgumentParser()
    parser.add_argument('--gpus', nargs='+', help='gpus to run on', required=True)
    parser.add_argument('--nb_experiments', type=int, required=True)
    parser.add_argument('--sampling_method', type=str, help='sampling method', required=True)
    args = parser.parse_args()

    gpu_busy = {}
    hparams = {}
    experiment_counter = 0

    gpus = args.gpus
    nb_experiments = args.nb_experiments
    sampling_method = get_sampling_method(args.sampling_method)

    for gpu in gpus:
        p = spawn_experiment(sampling_method, gpu, experiment_counter)
        gpu_busy[gpu] = p
        experiment_counter += 1

    while experiment_counter < nb_experiments:
        print("check if GPU available. {}/{}".format(experiment_counter+1, nb_experiments))
        time.sleep(60)
        for gpu in gpus:
            p = gpu_busy[gpu]
            if not p.poll() is None:
                p = spawn_experiment(sampling_method, gpu, experiment_counter)
                gpu_busy[gpu] = p
                experiment_counter += 1

if __name__ == '__main__':
    main()
