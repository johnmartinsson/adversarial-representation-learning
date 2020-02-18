import os
import pickle
import json
import numpy as np

def main():
    artifacts_dir = 'artifacts/filter_experiment/'
    artifacts_baseline_dir = 'artifacts/filter_baseline_experiment/'
    epsilons = ['0.001', '0.005', '0.01', '0.05']
    use_filters = ['True', 'False']

    table = {}
    table['f'] = []
    table['g'] = []
    table['g . f'] = []
    for use_filter in use_filters:
        for eps in epsilons:
            results_path = os.path.join(artifacts_dir,
                    'Smiling_eps_{}_use_filter_{}'.format(eps, use_filter), '0', 'results.json')

            with open(results_path, 'r') as f:
                res = json.load(f)

            if use_filter == 'True':
                table['g . f'].append(res['secret_adv_acc'] * 100.0)
            if use_filter == 'False':
                table['g'].append(res['secret_adv_acc'] * 100.0)

            if use_filter == 'True':
                results_baseline_path = os.path.join(artifacts_baseline_dir,
                        'Smiling_eps_{}_use_filter_{}'.format(eps, use_filter), '0', 'results.json')

                with open(results_baseline_path, 'r') as f:
                    res_baseline = json.load(f)

                table['f'].append(res_baseline['secret_adv_acc'] * 100.0)

    top_row_format = '{:>10} {:>10} {:>10} {:>10} {:>10}'
    row_format = '{:>10} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}'
    print(top_row_format.format("eps", *epsilons))
    for key in table.keys():
        #print(key)
        #print(table[key])
        print(row_format.format(key, *table[key]))
if __name__ == '__main__':
    main()
