import os
import pickle
import numpy as np
from scipy.stats import pearsonr

def main():
    artifacts_dir = 'artifacts/attributes_experiment/'
    secret_attrs = ['Smiling', 'Male', 'Wearing_Lipstick', 'Young']
    predicted_attrs = ['Smiling', 'Male', 'Wearing_Lipstick', 'Young',
    'High_Cheekbones', 'Mouth_Slightly_Open', 'Heavy_Makeup']

    table = {}
    for predicted_attr in predicted_attrs:
        table[predicted_attr] = []

        for secret_attr in secret_attrs:
            predict_path = 'predict_{}_64x64.pkl'.format(secret_attr)
            pkl_file_path = os.path.join(artifacts_dir, secret_attr + '_eps_0.005', '0', predict_path)

            with open(pkl_file_path, 'rb') as f:
                (main_preds, main_gen_preds, main_secrets, main_gen_secrets) = pickle.load(f)
                main_gen_preds = np.argmax(main_gen_preds, axis=1)

            acc = np.mean(main_gen_preds == main_gen_secrets)
            print(secret_attr, " : ", acc)

            #for predicted_attr in predicted_attrs:
            #    predict_path = 'predict_{}_64x64.pkl'.format(predicted_attr)
            #    pkl_file_path = os.path.join(artifacts_dir, secret_attr +
            #    '_eps_0.005', '0', predict_path)

            predict_path = 'predict_{}_64x64.pkl'.format(predicted_attr)
            pkl_file_path = os.path.join(artifacts_dir, secret_attr + '_eps_0.005', '0', predict_path)

            with open(pkl_file_path, 'rb') as f:
                (preds, gen_preds, secrets, gen_secrets) = pickle.load(f)
                preds     = np.argmax(preds, axis=1)
                gen_preds = np.argmax(gen_preds, axis=1)

            (pr, _)   = pearsonr(main_gen_preds, gen_preds)
            table[predicted_attr].append(pr)

    top_row_format = '{:>20} & {:>20} & {:>20} & {:>20} & {:>20} \\\\' # {:>20} {:>20} {:>20}'
    row_format = '{:>20} & {:>20.2f} & {:>20.2f} & {:>20.2f} & {:>20.2f} \\\\' # {:>20.2f} {:>20.2f} {:>20.2f}'
    print(top_row_format.format("", *secret_attrs))
    for key in table.keys():
        print(row_format.format(key, *table[key]))
if __name__ == '__main__':
    main()
