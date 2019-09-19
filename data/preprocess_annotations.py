import numpy as np
import pandas as pd

def load_annotation_file_as_dataframe(annotation_file):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    columns = ['filename'] + lines[1].strip().split(' ')
    data = []
    for l in lines[2:]:
        data.append(list(filter(lambda x: len(x) > 0, l.strip().split(' '))))
    df = pd.DataFrame(np.array(data), columns=columns)

    return df

def load_partition_file_as_dataframe(partition_file):
    with open(partition_file, 'r') as f:
        lines = f.readlines()
    columns = ['filename', 'subset']
    data = []

    for l in lines:
        data.append(l.strip().split(' '))
    df = pd.DataFrame(np.array(data), columns=columns)

    return df

def main():
    annotation_frame = load_annotation_file_as_dataframe('annotations.txt')
    data_split_frame = load_partition_file_as_dataframe('data_split.txt')
    nb_training_samples   = (data_split_frame['subset'] == '0').sum()
    nb_validation_samples = (data_split_frame['subset'] == '1').sum()
    nb_test_samples       = (data_split_frame['subset'] == '2').sum()

    train_idx_begin = 0
    train_idx_end   = nb_training_samples
    valid_idx_begin = train_idx_end
    valid_idx_end   = valid_idx_begin + nb_validation_samples
    test_idx_begin  = valid_idx_end
    test_idx_end    = test_idx_begin + nb_test_samples

    training_annotations_frame   = annotation_frame[train_idx_begin:train_idx_end]
    validation_annotations_frame = annotation_frame[valid_idx_begin:valid_idx_end]
    test_annotations_frame       = annotation_frame[test_idx_begin:test_idx_end]

    training_annotations_frame.to_csv('training_annotations.csv')
    validation_annotations_frame.to_csv('validation_annotations.csv')
    test_annotations_frame.to_csv('test_annotations.csv')

if __name__ == '__main__':
    main()
