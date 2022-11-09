# EdgeLearning

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Unix Status](https://github.com/mircodemarchi/EdgeLearning/workflows/Unix/badge.svg)](https://github.com/mircodemarchi/EdgeLearning/workflows/unix.yml)
[![Windows Status](https://github.com/mircodemarchi/EdgeLearning/workflows/Windows/badge.svg)](https://github.com/mircodemarchi/EdgeLearning/workflows/windows.yml)
[![codecov](https://codecov.io/gh/mircodemarchi/EdgeLearning/branch/main/graph/badge.svg)](https://codecov.io/gh/mircodemarchi/EdgeLearning)

EdgeLearning is a Deep Learning library for Edge Computing developed for high performance, low footprint on embedded devices and distributed systems.

The project library is supported on MacOS, Windows and Linux systems, with Clang and GNU GCC compiler.

## Fast setup

Commands to build the library with default configuration: 
```bash
git submodule update --init --recursive
mkdir build && cd build
cmake ..
cmake --build .
```

Commands to build the library with default configuration and Python interface:
```bash
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DBUILD_PYTHON=1
cmake --build .
```

Open the [example](./example) folder to get started and learn how to use the EdgeLearning library on known datasets.


## Dependencies

Clone the Git repository:
```bash
git clone https://github.com/mircodemarchi/EdgeLearning.git EdgeLearning
cd EdgeLearning
```

The Git repository has submodules, in order to download the submodules, it is necessary to issue: 

```bash
git submodule update --init --recursive
```


## Building

Setup the build subdirectory:
```bash
mkdir build && cd build
```

Then you need to prepare the build environment. 
Choose your desired configuration with the following parameters:

- With GNU GCC compiler: 
    ```bash
    cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 ..
    ```
- With Clang compiler:
    ```bash
    cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
    ```
- Enable [MLPACK](https://www.mlpack.org): the high level api of EdgeLearning allows to select [Mlpack](https://github.com/mlpack/mlpack) functionalities statically or dynamically. 
    ```bash
    cmake -DENABLE_MLPACK=1 ..
    ```
    N.B. The option ENABLE_MLPACK is not supported in MacOS with GNU GCC compiler.
  
- Build with Python interface:
    ```bash
    cmake -DBUILD_PYTHON=1 ..
    ```

Build and install the library:
```bash
make -j 
sudo make install
```

## Profiling

The profiling routines included in EdgeLearning explore different types of datasets by varying architecture and model parameters.

There are 4 profiling routines:
- Classification problem solved with Feedforward Neural Network: [profile_fnn_classification](./profile/profile_fnn_classification.cpp);
- Regression problem solved with Feedforward Neural Network: [profile_fnn_regression](./profile/profile_fnn_regression.cpp);
- Classification problem solved with Recurrent Neural Network: Work in Progress;
- Regression problem solved with Recurrent Neural Network: Work in Progress;

In order to execute the routines of profiling you need to compile the library as explained in [Building](#building) section. 
Then you can execute the profiling routines as following: 

```bash
./profile/profile_fnn_classification
./profile/profile_fnn_regression
```

The profiling routines will create a folder of the following pattern: profile_<framework>_<profile-type>, where framework could be "edgelearning" or "mlpack" based on the library configuration choice of the ENABLE_MLPACK field, and <profile-type> could be "fnn_classification", "fnn_regression", "rnn_classification" or "rnn_regression", based on the profiling routine executed.
The folder will contain all the performance measure produced by the execution of the profiling routine divided in files that represent the model architecture and parameter used in the profiling steps.


## Contribution guidelines

If you would like to contribute to Edge Learning, please contact the developer:

* Mirco De Marchi <mirco.demarchi@univr.it>
* Luca Geretti <luca.geretti@univr.it>
