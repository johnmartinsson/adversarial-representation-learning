import os
import pickle
import json
import numpy as np

def main():
    artifacts_dir = 'artifacts/attributes_experiment/'
    epsilons = ['0.001', '0.005', '0.01', '0.05']
    attributes = ['Smiling', 'Male', 'Wearing_Lipstick', 'Young']

    mean_table = {}
    std_table = {}
    for eps in epsilons:
        mean_table[eps] = []
        std_table[eps] = []
        for attr in attributes:
            gen_secret_accs = []
            for i in range(5):
                results_path = os.path.join(artifacts_dir,
                        '{}_eps_{}'.format(attr, eps), str(i), 'results.json')
                with open(results_path, 'r') as f:
                    res = json.load(f)
                gen_secret_accs.append(res['gen_secret_acc'] * 100)
            mean_table[eps].append(np.mean(gen_secret_accs))
            std_table[eps].append(np.std(gen_secret_accs))

    top_row_format = '{:>20} & {:>20} & {:>20} & {:>20} & {:>20} \\\\'
    row_format = '{:>20} & {:.1f} \pm {:.1f} & {:.1f} \pm {:.1f} & {:.1f} \pm {:.1f} & {:.1f} \pm {:.1f} \\\\'
    print(top_row_format.format("", *attributes))
    for key in mean_table.keys():
        row = list(np.concatenate(list(zip(mean_table[key], std_table[key]))))
        print(row_format.format(key, *row))
if __name__ == '__main__':
    main()
