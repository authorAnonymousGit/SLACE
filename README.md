# SLACE: A Monotone and Balance-Sensitive Loss Function for Ordinal Regression

## Introduction
Ordinal text classification is a special case oftext classification, in which there is an ordinal relationship among classes.
We suggest a novel loss function, WOCEL (weighted ordinal cross entropy loss), making use of information theory considerations and class ordinality.

## Prerequisites:  
1. GPU 
2. [Anaconda 3](https://www.anaconda.com/download/)  
3. [Pytorch](https://pytorch.org/)
4. [Transformers](https://pytorch.org/hub/huggingface_pytorch-transformers/)
5. [Keras (Version 2.4.3)](https://keras.io/)
6. The same as described in [SentiLARE](https://github.com/thu-coai/SentiLARE) (only if you are interested in running SentiLARE).


## Getting Started

## Datasets

| Dataset  | # Train | # Validation | # Test | # Labels |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| [SST-5](https://nlp.stanford.edu/sentiment/code.html)  | 8,544  | 1,101  |  2,210  | 5  | 
| [SemEval-2017 Task 4-A (English)](https://alt.qcri.org/semeval2017/task4/)  | 12,378  | 4,127  | 4,127  | 3  | 
| [Amazon (Amazon Electronics)](https://nijianmo.github.io/amazon/index.html)  | 8,998  | 2,998  | 2,999  | 5  | 

The used datasets are provided in the [data](./data/) folder, 
divided to train, validation and test.

Each file contains the following attributes:
* key_index: identifier.
* text
* overall: the correct class out of the possible labels.

## Running Experiments
1. Update the relevant config file ([Config0](./config0.py) or [Config1](./config1.py)). Use 0 if cuda=0 and 1 if cuda=1.
   with your configuration details. Please follow the details inside the given file.
2. Run [run_primary.sh](./run_pipeline.sh): 
