import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def scatterplot(data, label, point_labels, ax, color, style='-o'):
    ax.plot([x[0] for x in data], [x[1] for x in data], style, label=label,
            color=color)
    for i, txt in enumerate(point_labels):
        xy = (data[i][0]-0.9, data[i][1]+0.5)
        ax.annotate(r'$\epsilon = {}$'.format(txt), xy, fontsize=8)

def load_results(epsilons, ours_dir, baseline_dir, nb_runs, attr):
    adv_ours = []
    adv_bline = []

    for eps in epsilons:

        bline_adv_secret_acc = 0.0
        ours_adv_secret_acc = 0.0

        bline_mean_utility_acc = 0.0
        ours_mean_utility_acc = 0.0

        for i in range(nb_runs):
            ours_experiment_dir = os.path.join(ours_dir, '{}_eps_{}'.format(attr, eps), str(i))
            bline_experiment_dir = os.path.join(baseline_dir, '{}_eps_{}'.format(attr, eps), str(i))

            ours_results_path = os.path.join(ours_experiment_dir, 'results.json')
            bline_results_path = os.path.join(bline_experiment_dir, 'results.json')

            with open(ours_results_path, 'r') as f:
                ours_res = json.load(f)
            with open(bline_results_path, 'r') as f:
                bline_res = json.load(f)

            bline_adv_secret_acc += bline_res['secret_adv_acc']*100
            ours_adv_secret_acc += ours_res['secret_adv_acc']*100

            bline_mean_utility_acc += np.load(os.path.join(ours_experiment_dir, 'bline_mean_utility_acc.npy'))*100
            ours_mean_utility_acc  += np.load(os.path.join(ours_experiment_dir, 'ours_mean_utility_acc.npy'))*100

        adv_bline.append((bline_adv_secret_acc/nb_runs, bline_mean_utility_acc/nb_runs))
        adv_ours.append((ours_adv_secret_acc/nb_runs, ours_mean_utility_acc/nb_runs))

    return adv_ours, adv_bline

def main():
    artifacts_dir = 'artifacts/attributes_experiment/'
    baseline_dir = 'artifacts/attributes_baseline_experiment/'
    epsilons = ['0.001', '0.005', '0.01', '0.02', '0.03']
    use_filter = 'True'

    for attr in ['Smiling', 'Male', 'Wearing_Lipstick', 'Young']:
        if attr in ['Smiling', 'Male']: # entropy loss only run for two attributes
            adv_ours_ent, adv_bline_ent = load_results(epsilons,
                    'artifacts/attributes_entropy_experiment',
                    'artifacts/attributes_entropy_baseline_experiment', 2, attr)

        adv_ours, adv_bline = load_results(epsilons,
                'artifacts/attributes_experiment',
                'artifacts/attributes_baseline_experiment', 5, attr)

        plt.rcParams["mathtext.fontset"] = "cm"
        fig, ax = plt.subplots()

        scatterplot(adv_ours, label='ours (log-likelihood)', point_labels=epsilons, ax=ax,
                color='green')
        scatterplot(adv_bline, label='baseline (log-likelihood)', point_labels=epsilons, ax=ax,
                color='red')

        if attr in ['Smiling', 'Male']: # ablation only run for two attributes
            scatterplot(adv_ours_ent, label='ours (entropy)', point_labels=epsilons,
                    ax=ax, color='green', style='--x')
            scatterplot(adv_bline_ent, label='baseline (entropy)',
                    point_labels=epsilons, ax=ax, color='red', style='--x')
        
        if attr == 'Wearing_Lipstick':
            plt.title('Lipstick')
        elif attr == 'Young':
            plt.title('Age')
        else:
            plt.title(attr)
        plt.ylabel("Utility score")
        plt.xlabel("Privacy loss".format(attr.lower()))
        plt.legend(loc='upper left')
        plt.savefig('{}_privacy_utility_tradeoff.pdf'.format(attr))

if __name__ == '__main__':
    main()
