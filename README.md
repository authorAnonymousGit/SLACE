# SLACE: A Monotone and Balance-Sensitive Loss Function for Ordinal Regression

## Introduction
Ordinal classification is a special case of classification, in which there is an ordinal relationship among classes.
We suggest a novel loss function, SLACE (Soft Labels Accumulating Cross Entropy), making use of information theory considerations and class ordinality.

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


## Files
1. [SLACE_full_paper.pdf](./SLACE_full_paper.pdf): SLACE paper cointaining full proofs.
2. [Config.py](./config.py): Contains what datasets we want to use and the devision of the data, all datasets we used and their devisions are in comment,
loss functions we want to check, alphas, range of random seeds, and params for the xgboost classifier.
3. [Process_data.py](./process_data.py): Functions for processing of datasets and binning.
4. [Losses.py](./losses.py): Implementation of all loss functions mentioned in [paper](./SLACE_full_paper.pdf).
5. [Utils.py](./utils.py): Implementation of prox, prox dom, and CEM.
6. [Main.py](./main.py): Running the xgb classifier by config. Saves each xgb classifier prediction in csvs with columns Predictions,Probabilities, and Labels.
7. Files in [code_for_graphs_from_paper](./code_for_graphs_from_paper/) contain the code to analyze the outputs from main to create the graphs in the paper.

## Experiments
1. Run [Main.py](./main.py).
2. Analyze results with [code_for_graphs_from_paper](./code_for_graphs_from_paper/).
