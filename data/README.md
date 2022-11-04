# Datasets folder

Download here the datasets that you are going to use for the library validation and profiling. 


## [MNIST](http://yann.lecun.com/exdb/mnist/)

Click [here](http://yann.lecun.com/exdb/mnist/) and download the MNIST dataset and put the files in the [data](.) folder. 
The files that the profiling functionalities expects are the following:

- [train-images.idx3-ubyte](./train-images.idx3-ubyte)
- [train-labels.idx1-ubyte](./train-labels.idx1-ubyte)
- [t10k-images.idx3-ubyte](./t10k-images.idx3-ubyte)
- [t10k-labels.idx1-ubyte](./t10k-labels.idx1-ubyte)

The same requirements are expected also for [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) since the format is the same. 


## [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

Click [here](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) to download the zip that contains the Cifar-10 dataset in binary version and put the files in the [data](.) folder.
The files that the profiling functionalities expects are the following:

- [data_batch_1.bin](./data_batch_1.bin)
- [data_batch_2.bin](./data_batch_2.bin)
- [data_batch_3.bin](./data_batch_3.bin)
- [data_batch_4.bin](./data_batch_4.bin)
- [data_batch_5.bin](./data_batch_5.bin)
- [test_batch.bin](./test_batch.bin)
- [batches.meta.txt](./batches.meta.txt)


## [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

Click [here](https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz) to download the zip that contains the Cifar-100 dataset in binary version and put the files in the [data](.) folder.
The files that the profiling functionalities expects are the following:

- [train.bin](./train.bin)
- [test.bin](./test.bin)
- [coarse_label_names.txt](./coarse_label_names.txt)
- [fine_label_names.txt](./fine_label_names.txt)


## [MLPACK Thyroid](https://github.com/mlpack/mlpack/tree/master/src/mlpack/tests/data)

Download the [thyroid_train.csv](https://github.com/mlpack/mlpack/blob/master/src/mlpack/tests/data/thyroid_train.csv) and the [thyroid_test.csv](https://github.com/mlpack/mlpack/blob/master/src/mlpack/tests/data/thyroid_train.csv) files from the main repository of the Mlpack machine learning framework.
The dataset is provided by Mlpack for their model accuracy tests in CSV format. 
The files that the profiling functionalities expects are the following:

- [thyroid_train.csv](./thyroid_train.csv)
- [thyroid_test.csv](./thyroid_test.csv)


## Execution Time

The Execution time dataset is a dataset for performance measure forecasting of concurrent tasks that is already provided in the project in the CSV format.
The dataset provides a regression problem and it is contained in the following file: 

- [execution-time.csv](./execution-time.csv)

