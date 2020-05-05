import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def scatterplot(attribute_idx, table, epsilons, marker, color, label, ax):
    ys = []
    xs = []
    for eps in epsilons:
        row = table[eps]
        y = row[attribute_idx]
        ys.append(y)
        xs.append(float(eps))
    ax.scatter(x=xs, y=ys, marker=marker, color=color, label=label)

def main():
    artifacts_dir = 'artifacts/attributes_experiment/'
    baseline_dir  = 'artifacts/attributes_baseline_experiment/'
    epsilons = ['0.001', '0.005', '0.01', '0.05']
    attributes = ['Male', 'Wearing_Lipstick', 'Young']

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

    colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, ax = plt.subplots()
    for i, attr in enumerate(attributes):
        scatterplot(i, mean_table_ours, epsilons, marker='o', color=colors[i],
                label="{}".format(attr), ax=ax)
        scatterplot(i, mean_table_bline, epsilons, marker='x', color=colors[i],
                label="".format(attr), ax=ax)
    plt.legend(loc='lower left')
    plt.ylabel("Adv. smiling accuracy [%]")
    plt.xlabel(r'$\epsilon$')
    plt.show()

if __name__ == '__main__':
    main()
