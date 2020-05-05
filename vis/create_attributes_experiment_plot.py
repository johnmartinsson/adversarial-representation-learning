import os
import pickle
import json
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def scatterplot(attribute_idx, table, epsilons, label, ax):
    ys = []
    xs = []
    for eps in epsilons:
        row = table[eps]
        y = row[attribute_idx]
        ys.append(y)
        xs.append(float(eps))
    ax.scatter(x=xs, y=ys, label=label)


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

    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots()
    for i, attr in enumerate(attributes):
        scatterplot(i, mean_table, epsilons, label=attr, ax=ax)

    plt.ylabel("Fool fixed classifier [%]")
    plt.xlabel("$\epsilon$")
    plt.legend(loc="lower right")
    plt.savefig("fool_fixed_classifier.pdf")

if __name__ == '__main__':
    main()
