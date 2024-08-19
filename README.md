# SLACE: A Monotone and Balance-Sensitive Loss Function for Ordinal Regression

## Introduction
Ordinal text classification is a special case oftext classification, in which there is an ordinal relationship among classes.
We suggest a novel loss function, WOCEL (weighted ordinal cross entropy loss), making use of information theory considerations and class ordinality.

## Prerequisites:  
1. Python (3.6.8)
2. [Pytorch](https://pytorch.org/)(1.10.2) 
3. pandas (1.1.5)
4. numpy (1.19.5)
5. sklearn (0.24.2)
6. scipy (1.5.4)
7. xgboost (1.5.2)


## Getting Started

## Datasets
The used datasets are provided in the [data](./datasets/) folder.


## Running Experiments
1. Update the relevant config file ([Config0](./config0.py) or [Config1](./config1.py)). Use 0 if cuda=0 and 1 if cuda=1.
   with your configuration details. Please follow the details inside the given file.
2. Run [run_primary.sh](./run_pipeline.sh): 
