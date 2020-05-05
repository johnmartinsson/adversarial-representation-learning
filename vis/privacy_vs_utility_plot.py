import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def scatterplot(data, label, point_labels, ax):
    ax.scatter(x=[x[0] for x in data], y=[x[1] for x in data], label=label)
    for i, txt in enumerate(point_labels):
        ax.annotate(r'$\epsilon = {}$'.format(txt), (data[i][0]-1.0,
            data[i][1]+0.7), fontsize=8)

def main():
    artifacts_dir = 'artifacts/attributes_experiment/'
    baseline_dir = 'artifacts/attributes_baseline_experiment/'
    epsilons = ['0.001', '0.005', '0.01', '0.05']
    use_filter = 'True'
    nb_runs = 5

    fix_ours = []
    fix_bline = []
    adv_ours = []
    adv_bline = []

    for eps in epsilons:

        bline_fix_secret_acc = 0.0
        ours_fix_secret_acc = 0.0
        bline_adv_secret_acc = 0.0
        ours_adv_secret_acc = 0.0
        bline_fix_utility_acc = 0.0
        ours_fix_utility_acc = 0.0

        for i in range(nb_runs):
            ours_results_path = os.path.join(artifacts_dir,
                    'Smiling_eps_{}'.format(eps), str(i), 'results.json')
            bline_results_path = os.path.join(baseline_dir,
                    'Smiling_eps_{}'.format(eps), str(i), 'results.json')
            with open(ours_results_path, 'r') as f:
                ours_res = json.load(f)
            with open(bline_results_path, 'r') as f:
                bline_res = json.load(f)

            bline_fix_secret_acc += bline_res['fix_secret_acc']*100
            ours_fix_secret_acc += ours_res['fix_secret_acc']*100

            bline_adv_secret_acc += bline_res['secret_adv_acc']*100
            ours_adv_secret_acc += ours_res['secret_adv_acc']*100

            bline_fix_utility_acc += bline_res['fix_utility_acc']*100
            ours_fix_utility_acc  += ours_res['fix_utility_acc']*100
            #bline_fix_utility_acc += bline_res['fid']
            #ours_fix_utility_acc  += ours_res['fid']

        fix_bline.append((bline_fix_secret_acc/nb_runs, bline_fix_utility_acc/nb_runs))
        fix_ours.append((ours_fix_secret_acc/nb_runs, ours_fix_utility_acc/nb_runs))
        adv_bline.append((bline_adv_secret_acc/nb_runs, bline_fix_utility_acc/nb_runs))
        adv_ours.append((ours_adv_secret_acc/nb_runs, ours_fix_utility_acc/nb_runs))

    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots()

    scatterplot(adv_ours, label='ours', point_labels=epsilons, ax=ax)
    scatterplot(adv_bline, label='baseline', point_labels=epsilons, ax=ax)
    #scatterplot(fix_ours, label='fix. ours', point_labels=epsilons, ax=ax)
    #scatterplot(fix_bline, label='fix. bline', point_labels=epsilons, ax=ax)
    
    plt.xlabel("Adv. smiling accuracy [%]")
    #plt.ylabel(r'Frechet Inception distance')
    #plt.legend(loc='upper right')

    plt.ylabel("Fix. gender accuracy [%]")
    plt.legend(loc='lower right')
    plt.savefig('privacy_utility_tradeoff.pdf')

if __name__ == '__main__':
    main()
