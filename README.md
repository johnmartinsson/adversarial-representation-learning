# Instructions to train the private conditional generative adversarial network (PCGAN)
In this README we include a detailed description on how to download the data
used in this study, how to setup the environment to run the scripts in, and how
to run the experiments and produce results.


## File structure

	.
	├── artifacts				# training results root path
	│   └── fixed_classifiers               # experiments path
	│       └── Smiling                     # experiment result path
	│           ├── classifier_64x64.h5     # experiment result (e.g., model weights)
	│           .
	├── cgan.py
	├── data
	│   ├── doit.sh				# data download script
	│   ├── get_drive_file.py
	│   ├── __ini__.py
	│   ├── preprocess_annotations.py
	├── datasets
	│   └── celeba.py			# dataset definition
	├── environment.yml			# anaconda environment file
	├── loss_functions.py
	├── models
	│   ├── decoder.py
	│   ├── discriminator.py
	│   ├── encoder.py
	│   ├── filter.py
	│   ├── generator.py
	│   ├── inception.py
	│   ├── __init__.py
	│   ├── unet.py				# UNet architecture definition
	│   └── utils.py
	├── pcgan.py				# main implementation script
	├── README.md      
	├── run_experiment.py                   # experiment(s) main script
	├── sanity_check.py
	├── train_classifiers.py                # fixed classifier training script
	├── utils.py
	└── vis 				# table / plot scripts
	    ├── create_adversarial_table.py
	    .

## Setup environment (anaconda)
Install anaconda.

	conda env create -f environment.yml
	conda activate pcgan

## Download data
To download the data simply run the following commands:

	cd data
	sh doit.sh

We have noticed that the urls poitning to the Google Drive that hosts the data
can exceeed their quota. If the script does not work it is probably due to
this. Try downloading the files manually from the Google Drive or if that does not work the Baidu drive

	cd data
	download imgs_aligned.zip and unzip as imgs
	download list_eval_partition.txt as data_split.txt
	download list_attr_celeba.txt as annotations.txt
	python3 preprocess_annotations.py
	cd ..
	python3 data/preprocess_images.py

all these files need to be in the folder './data' with the correct names before running the preprocessing script.

You should now have

	.
	├── data
	│   ├── .
	│   ├── imgs
	│   ├── training_annotations.txt 	# annotations
	│   ├── validation_annotations.txt
	│   ├── test_annotations.txt
	│   ├── celeba_images_train_64x64.npy   # images in numpy format
	│   ├── celeba_images_valid_64x64.npy
	│   ├── celeba_images_test_64x64.npy

in the data folder. Next run

	python sanity_check.py

and open the resulting 'sanity_check.png' image to convince yourself that the
data is loaded properly.


## Train the models
List the number of GPUs you want to use to run the experiment. In the examples
the experiments will run on GPU 0 and GPU 1.

	python run_experiment.py --gpus 0 1 --experiment_name=train_classifiers --mode=train
	python run_experiment.py --gpus 0 1 --experiment_name=attributes_experiment --mode=train
	python run_experiment.py --gpus 0 1 --experiment_name=filter_experiment --mode=train

## Evaluate the models
To evaluate the main method experiment:

	python run_experiment.py --gpus 0 1 --experiment_name=attributes_experiment --mode=evaluate
	python run_experiment.py --gpus 0 1 --experiment_name=filter_experiment --mode=evaluate

To evaluate the baseline. Since the update of the filter model is independent
of the generator we can simply copy the weights of the filter model from the
main experiment and use them to evaluate the baseline. The main difference is
that we ONLY run the images through the filter in this evaluation.

	cp -r artifacts/attributes_experiment artifacts/attributes_baseline_experiment
	python run_experiment.py --gpus 0 1 --experiment_name=attributes_baseline_experiment --mode=evaluate

## Visualize the output of the models
To visualize the output of the models run:

	python run_experiment.py --gpus 0 1 --experiment_name=attributes_experiment --mode=visualize

and check the 

	artifacts/attributes_experiment/visualization

folder for the images.

## Produce main tables

	python vis/create_adversarial_table.py			# Table TODO
	python vis/create_attributes_experiment_table.py        # Table TODO
	python vis/create_fix_classifier_table.py               # Table TODO
	python vis/create_filter_experiment_table.py            # Table TODO

## Inspect the loss-functions during training
To visualize the loss-functions and validation functions during training

	tensorboard --logdir=artifacts/<experiment_path>

