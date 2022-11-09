# EdgeLearning Examples

This folder contains various examples of training models on known datasets with EdgeLearning. Based on the ENABLE_MLPACK configuration flag, the high-level api will perform operations with [Mlpack framework](https://github.com/mlpack/mlpack).

In order to execute the examples you need to download the desired dataset as described in the [data](../data) folder. 

A brief description of the examples in this folder follows: 


- [simple_classification](./simple_classification.cpp): 

Create a simple model with 2 hidden layers trained on dummy data allocated with randomly values in the range [-1,+1] with an user defined classification function that construct the labels.

- [simple_regression](./simple_regression.cpp):

Create a simple model with 2 hidden layers trained on dummy data allocated with randomly values in the range [-1,+1] with a set of user defined functions that represent the result to predict.

- []
