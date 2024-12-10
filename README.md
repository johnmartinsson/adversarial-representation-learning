# Adversarial representation learning for synthetic replacement of private attributes
The official implementation of the paper "Adversarial representation learning for synthetic replacement of private attributes".

[Paper](https://johnmartinsson.org/publications/2021/adversarial-representation-learning-images)

Cite as:

    @article{Martinsson2021,
      author = {Martinsson, John and Zec, Edvin Listo and Gillblad, Daniel and Mogren, Olof},
      title = {Adversarial representation learning for synthetic replacement of private attributes},
      journal = {Proceedings - 2021 IEEE International Conference on Big Data, Big Data 2021},
      year = {2021},
      pages = {1291--1299},
      doi = {10.1109/BigData52589.2021.9671802},
      url = {https://ieeexplore.ieee.org/document/9671802}
    }

# Instructions for learning synthetic replacement of private attributes
In this README we include a detailed description on how to download the data
used in this study, how to setup the environment to run the scripts in, and how
to run the experiments and produce results.

## File structure

    .
    ├── artifacts                           # training results root path
    │   └── fixed_classifiers               # experiments path
    │       └── Smiling                     # experiment result path
    │           ├── classifier_64x64.h5     # experiment result (e.g., model weights)
    │           .
    ├── cgan.py
    ├── data
    │   ├── doit.sh                         # data rename / extraction script
    │   ├── __ini__.py
    │   ├── preprocess_annotations.py       # create csv files for data splits
    │   ├── preprocess_images.py            # create numpy files for images (trade space for speed)
    ├── datasets
    │   └── celeba.py                       # dataset definition
    ├── environment.yml                     # anaconda environment file
    ├── loss_functions.py
    ├── models
    │   ├── inception.py                    # used to compute FID
    │   ├── __init__.py
    │   ├── unet.py                         # UNet architecture definition
    │   └── utils.py
    ├── pcgan.py                            # main implementation script
    ├── README.md      
    ├── run_experiment.py                   # experiment(s) main script
    ├── sanity_check.py
    ├── train_classifiers.py                # fixed classifier training script
    ├── utils.py
    └── vis                                 # table / plot scripts
        ├── create_adversarial_table.py
        .

## Setup environment (anaconda)
Install anaconda.

    conda env create -f environment.yml
    conda activate pcgan

## Download data
Download each of these files from the Google Drive:

- [list_attr_celeba.txt](https://drive.google.com/open?id=0B7EVK8r0v71pblRyaVFSWGxPY0U),
- [list_eval_partition.txt](https://drive.google.com/open?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk),
- [img_align_celeba.zip](https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM),

and put them in the folder './data'. 

    cd data
    sh doit.sh # rename files and extracts zip
    python3 preprocess_annotations.py

You should now have

    .
    ├── data
    │   ├── .
    │   ├── imgs
    │   ├── training_annotations.txt  # annotations
    │   ├── validation_annotations.txt
    │   ├── test_annotations.txt

in the data folder. 

Next run the sanity check

    python sanity_check.py

and open the resulting 'sanity_check.png' image to convince yourself that the
data is loaded properly.

## Demo experiment
Run a short experiment to produce a trade-off curve plot for the __smiling__ attribute with an average over __one__ run. Remember, evaluation requires training of an adversary to predict the sensitive attribute for each configuration, and can take some time.

    # train the fixed classifiers
    python run_experiment.py --gpus 0 1 --experiment_name=train_classifiers --mode=train

    # train and evaluate models
    python run_experiment.py --gpus 0 1 --experiment_name=demo_experiment --mode=train
    python run_experiment.py --gpus 0 1 --experiment_name=demo_experiment --mode=evaluate
    python run_experiment.py --gpus 0 1 --experiment_name=demo_experiment --mode=predict_utility
    
    # evaluate model and only run images through the filter
    cp -r artifacts/demo_experiment/ artifacts/demo_baseline_experiment/
    python run_experiment.py --gpus 0 1 --experiment_name=demo_baseline_experiment --mode=evaluate
    python run_experiment.py --gpus 0 1 --experiment_name=demo_baseline_experiment --mode=predict_utility

    # create trade-off curve
    python vis/privacy_vs_utility_demo_plot.py
    
The trade-off curve is saved in the working directory as 'Smiling_privacy_utility_tradeoff.pdf'.

## Train the models
List the number of GPUs you want to use to run the experiment. In the examples
the experiments will run on GPU 0 and GPU 1.

    python run_experiment.py --gpus 0 1 --experiment_name=train_classifiers --mode=train
    python run_experiment.py --gpus 0 1 --experiment_name=attributes_experiment --mode=train
    python run_experiment.py --gpus 0 1 --experiment_name=attributes_entropy_experiment --mode=train
    python run_experiment.py --gpus 0 1 --experiment_name=filter_experiment --mode=train

## Evaluate the models
To evaluate the main method experiment:

    python run_experiment.py --gpus 0 1 --experiment_name=attributes_experiment --mode=evaluate
    python run_experiment.py --gpus 0 1 --experiment_name=attributes_experiment --mode=predict_utility
    python run_experiment.py --gpus 0 1 --experiment_name=attributes_entropy_experiment --mode=evaluate
    python run_experiment.py --gpus 0 1 --experiment_name=attributes_entropy_experiment --mode=predict_utility
    python run_experiment.py --gpus 0 1 --experiment_name=filter_experiment --mode=evaluate

To evaluate the baseline. Since the update of the filter model is independent
of the generator we can simply copy the weights of the filter model from the
main experiment and use them to evaluate the baseline. The difference is
that we ONLY run the images through the filter in this evaluation.

    cp -r artifacts/attributes_experiment artifacts/attributes_baseline_experiment
    python run_experiment.py --gpus 0 1 --experiment_name=attributes_baseline_experiment --mode=evaluate
    python run_experiment.py --gpus 0 1 --experiment_name=attributes_baseline_experiment --mode=predict_utility
    cp -r artifacts/attributes_entropy_experiment artifacts/attributes_entropy_baseline_experiment
    python run_experiment.py --gpus 0 1 --experiment_name=attributes_entropy_baseline_experiment --mode=evaluate
    python run_experiment.py --gpus 0 1 --experiment_name=attributes_entropy_baseline_experiment --mode=predict_utility
    cp -r artifacts/filter_experiment artifacts/filter_baseline_experiment
    python run_experiment.py --gpus 0 1 --experiment_name=filter_baseline_experiment --mode=evaluate

## Produce main figure

### Figure 1
The trade-off between utility score and privacy loss for different distortion budgets. This will create the four plots seen in Figure 1 in the paper saved as pdfs in the working directory.

    python vis/privacy_vs_utility_plot.py

## Produce main tables

### Table 1
The success rate of our method to fool the fixed classifier that the synthetic
sensitive attribute is in the censored image.

    python vis/create_attributes_experiment_table.py

### Table 2
The value of each cell denotes the Pearson’s correlation coefficient between
predictions from a fixed classifier trained to predict the row attribute and a
fixed classifier trained to predict the column attribute, given that the column
attribute has been censored.

    # need to run additional predictions
    python run_experiment.py --gpus 0 1 --experiment_name=attributes_experiment --mode=predict
    # create table
    python vis/create_correlation_table.py

### Table 3
The results of evaluating the adversarially trained classifiers on the held out
test data censored with the baseline, only the generator, and our method.

    python vis/create_filter_experiment_table.py

(Output table is basically transposed w.r.t table in paper.)

## Visualize the output of the models
To visualize the output of the models run:

    python run_experiment.py --gpus 0 1 --experiment_name=attributes_experiment --mode=visualize

and check the 

    artifacts/attributes_experiment/visualization

folder for the images.


## Inspect the loss-functions during training
To visualize the loss-functions and validation functions during training

    tensorboard --logdir=artifacts/<experiment_path>

