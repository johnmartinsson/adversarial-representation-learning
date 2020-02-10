# Instructions to train the private conditional generative adversarial network (PCGAN)
In this README we include a detailed description on how to download the data
used in this study, how to setup the environment to run the scripts in, and how
to run the experiments and produce results.

## Download data
To download the data simply run the following commands:

	cd data
	sh doit.sh

We have notived that the urls poitning to the Google Drive that hosts the data
can exceeed their quota from time to time. If the script does not work it is
probably due to this. Simply check back at a later time.

## Setup environment (anaconda)
Install anaconda.

	conda env create -f environment.yml
	conda activate pcgan


## Train the models

	python run_experiment.py --gpus 0 1 --experiment_name train_classifiers --mode train

## Evaluate the models

## Visualize the output of the models

## Download fixed classifiers
	
	cd classifiers
	sh doit.sh
