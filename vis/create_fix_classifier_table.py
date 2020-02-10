import os
import pickle
import json
import numpy as np

def main():
    artifacts_dir = 'artifacts/attributes_experiment/'
    baseline_dir = 'artifacts/attributes_baseline_experiment/'
    epsilons = ['0.001', '0.005', '0.01', '0.05']
    use_filter = 'True'

    mean_table = {}
    std_table = {}
    for eps in epsilons:
        mean_table[eps] = []
        std_table[eps] = []

        bline_fix_secret_accs = []
        ours_fix_secret_accs = []
        bline_fix_utility_accs = []
        ours_fix_utility_accs = []
        bline_fids = []
        ours_fids = []

        for i in range(5):
            ours_results_path = os.path.join(artifacts_dir,
                    'Smiling_eps_{}'.format(eps), str(i), 'results.json')
            bline_results_path = os.path.join(baseline_dir,
                    'Smiling_eps_{}'.format(eps), str(i), 'results.json')
            with open(ours_results_path, 'r') as f:
                ours_res = json.load(f)
            with open(bline_results_path, 'r') as f:
                bline_res = json.load(f)
            bline_fix_secret_acc = bline_res['fix_secret_acc']*100
            #if bline_fix_secret_acc <= 50.0:
            #    bline_fix_secret_acc = 100.0-bline_fix_secret_acc
            ours_fix_secret_acc = ours_res['fix_secret_acc']*100
            #if ours_fix_secret_acc <= 50.0:
            #    ours_fix_secret_acc = 100.0-ours_fix_secret_acc


            bline_fix_secret_accs.append(bline_fix_secret_acc)
            ours_fix_secret_accs.append(ours_fix_secret_acc)
            bline_fix_utility_accs.append(bline_res['fix_utility_acc']*100)
            ours_fix_utility_accs.append(ours_res['fix_utility_acc']*100)
            bline_fids.append(bline_res['fid'])
            ours_fids.append(ours_res['fid'])

        mean_table[eps].append(np.mean(bline_fix_secret_accs))
        mean_table[eps].append(np.mean(ours_fix_secret_accs))
        mean_table[eps].append(np.mean(bline_fix_utility_accs))
        mean_table[eps].append(np.mean(ours_fix_utility_accs))
        mean_table[eps].append(np.mean(bline_fids))
        mean_table[eps].append(np.mean(ours_fids))

        std_table[eps].append(np.std(bline_fix_secret_accs))
        std_table[eps].append(np.std(ours_fix_secret_accs))
        std_table[eps].append(np.std(bline_fix_utility_accs))
        std_table[eps].append(np.std(ours_fix_utility_accs))
        std_table[eps].append(np.std(bline_fids))
        std_table[eps].append(np.std(ours_fids))

    #top_row_format = '{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}'
    #row_format = '{:>10} & {:.1f} & {:.1f} \pm {:.1f} & {:.1f} \pm {:.1f} & {:.1f} \pm {:.1f} & {:.1f} \pm {:.1f} & {:.1f} \pm {:.1f} \\\\'
    row_format_1 = '{:>10} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} \\\\'
    row_format_2 = '{:>10} & \pm {:.1f} & \pm {:.1f} & \pm {:.1f} & \pm {:.1f} & \pm {:.1f} & \pm {:.1f} \\\\'
    #print(top_row_format.format("", *epsilons))
    for key in mean_table.keys():
        #row = list(np.concatenate(list(zip(mean_table[key], std_table[key]))))
        print(row_format_1.format(key, *mean_table[key]))
        print(row_format_2.format('', *std_table[key]))
if __name__ == '__main__':
    main()
