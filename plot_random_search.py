import os
from argparse import ArgumentParser
import glob
import json
import matplotlib.pyplot as plt

def main():
    parser = ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--y_axis', type=str, required=True)
    parser.add_argument('--x_axis', type=str, required=True)
    args = parser.parse_args()

    experiment_paths = glob.glob(os.path.join(args.experiment_dir, '*'))
    x_values = []
    y_values = []
    for experiment_path in experiment_paths:
        with open(os.path.join(experiment_path, 'evaluation_results.json'), 'r') as f:
            res = json.load(f)
        with open(os.path.join(experiment_path, 'hparams.json'), 'r') as f:
            hparams = json.load(f)

        y_value = res[args.y_axis]
        x_value = hparams[args.x_axis]
        x_values.append(x_value)
        y_values.append(y_value)

    plt.scatter(x_value, y_value)
    plt.xlabel(args.x_axis)
    plt.ylabel(args.y_axis)
    plt.show()

if __name__ == '__main__':
    main()
