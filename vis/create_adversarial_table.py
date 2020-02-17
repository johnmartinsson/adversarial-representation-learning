import os
import pickle
import json
import numpy as np

def main():
    artifacts_dir = 'artifacts/attributes_experiment/'
    baseline_dir  = 'artifacts/attributes_baseline_experiment/'
    epsilons = ['0.001', '0.005', '0.01', '0.05']
    attributes = ['Smiling', 'Male', 'Wearing_Lipstick', 'Young']

    mean_table_ours = {}
    mean_table_bline = {}
    for eps in epsilons:
        mean_table_ours[eps] = []
        mean_table_bline[eps] = []
        for attr in attributes:
            ours_adv_secret_accs = []
            bline_adv_secret_accs = []
            for i in range(5):
                ours_path = os.path.join(artifacts_dir,
                        '{}_eps_{}'.format(attr, eps), str(i), 'results.json')
                bline_path = os.path.join(baseline_dir,
                        '{}_eps_{}'.format(attr, eps), str(i), 'results.json')
                with open(ours_path, 'r') as f:
                    ours_res = json.load(f)
                with open(bline_path, 'r') as f:
                    bline_res = json.load(f)

                ours_adv_secret_accs.append(ours_res['secret_adv_acc'] * 100)
                bline_adv_secret_accs.append(bline_res['secret_adv_acc'] * 100)

            mean_table_ours[eps].append(np.mean(ours_adv_secret_accs))
            mean_table_bline[eps].append(np.mean(bline_adv_secret_accs))

    top_row_format = '{:>20} & {:>20} & {:>20} & {:>20} & {:>20} \\\\'
    row_format = '{:>20} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\'
    row_format_str = '{:>20} & {} & {} & {} & {} & {} & {} & {} & {} \\\\'

    print(top_row_format.format("", *attributes))
    print(row_format_str.format('epsilon', 'b-line', 'ours', 'b-line', 'ours',
        'b-line', 'ours', 'b-line', 'ours'))
    for key in mean_table_ours.keys():
        row = list(np.concatenate(list(zip(mean_table_bline[key], mean_table_ours[key]))))
        print(row_format.format(key, *row))
if __name__ == '__main__':
    main()
