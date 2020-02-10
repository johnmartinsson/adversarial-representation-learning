import os
import pickle
import json
import numpy as np

def main():
    artifacts_dir = 'artifacts/filter_experiment/'
    epsilons = ['0.001', '0.005', '0.01', '0.05']
    use_filters = ['True', 'False']

    table = {}
    for use_filter in use_filters:
        table[use_filter] = []
        for eps in epsilons:
            results_path = os.path.join(artifacts_dir,
                    'Smiling_eps_{}_use_filter_{}'.format(eps, use_filter), '0', 'results.json')
            with open(results_path, 'r') as f:
                res = json.load(f)
            table[use_filter].append(res['secret_adv_acc'])

    top_row_format = '{:>10} {:>10} {:>10} {:>10} {:>10}'
    row_format = '{:>10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}'
    print(top_row_format.format("", *epsilons))
    for key in table.keys():
        #print(key)
        #print(table[key])
        print(row_format.format(key, *table[key]))
if __name__ == '__main__':
    main()
