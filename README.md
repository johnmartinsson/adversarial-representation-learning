# Instructions to train model

## Download data

	cd data
	sh doit.sh

# Instructions to reproduce results



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
