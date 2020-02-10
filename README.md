# Instructions to train model

## Download data

	cd data
	sh doit.sh

# Instructions to reproduce results

## Setup environment (anaconda)

	conda create --name env python=3.8
	conda activate env
	conda install tqdm



## Train the models

	python run_experiment.py --gpus 0 1 --experiment_name train_classifiers --mode train

	

## Evaluate the models

## Visualize the output of the models



# Relevant files

pcgan.py

# Download data

	cd data
	sh doit.sh

# Fixed classifiers
The fixed classifiers for the secret (smiling) and the utility (gender) can
either be downloaded from a google drive, or be trained from scratch by
following the instructions below.

## Download fixed classifiers
	
	cd classifiers
	sh doit.sh

## Train fixed classifiers

	python3 train_classifiers.py

# Train adversarial bottleneck model
