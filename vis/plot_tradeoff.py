import numpy as np
import os
import argparse
import tensorflow as tf
import glob

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar_names", nargs='+', required=True)
    parser.add_argument("-l", "--experiment_paths", nargs='+', required=True)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    args = parser.parse_args()

    d = {}
    for experiment_path in args.experiment_paths:
        d[experiment_path] = {}
        for scalar_name in args.scalar_names:
            d[experiment_path][scalar_name] = []

    for experiment_path in args.experiment_paths:
        event_files = glob.glob(os.path.join(experiment_path, 'events.out.tfevents.*'))
        event_files = sorted(event_files)

        for event_file in event_files:
            print(event_file)
            for e in tf.train.summary_iterator(event_file):
                for v in e.summary.value:
                    for scalar_name in args.scalar_names:
                        if scalar_name in v.tag:
                            d[experiment_path][scalar_name].append(v.simple_value)

    for ep in d.keys():
        sn1 = list(d[ep].keys())[0]
        sn2 = list(d[ep].keys())[1]
        values1 = np.array(d[ep][sn1])
        values2 = np.array(d[ep][sn2])
        values  = np.abs(values1 - values2)
        plt.plot(values, label=ep[-10:])

    plt.ylabel('Tradeoff')
    plt.xlabel('Epoch')
    plt.legend(loc='lower left')
    plt.savefig(args.output_file)

if __name__ == '__main__':
    main()
