# EdgeLearning
Deep Learning for Edge Computing

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Unix Status](https://github.com/mircodemarchi/EdgeLearning/workflows/Unix/badge.svg)](https://github.com/mircodemarchi/EdgeLearning/workflows/unix.yml)
[![Windows Status](https://github.com/mircodemarchi/EdgeLearning/workflows/Windows/badge.svg)](https://github.com/mircodemarchi/EdgeLearning/workflows/windows.yml)
[![codecov](https://codecov.io/gh/mircodemarchi/EdgeLearning/branch/main/graph/badge.svg)](https://codecov.io/gh/mircodemarchi/EdgeLearning)

The project library is supported on MacOS and Linux systems, with Clang and GNU GCC compiler. 

N.B. The option ENABLE_MLPACK at true is not supported in MacOS with GNU GCC compiler. 

## Configuration 

The Git repository has submodules. In order to clone the submodules, it is necessary to issue

```
$ git submodule update --init --recursive
```

- With GNU GCC compiler: 
    ```
    $ cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 ..
    ```
- With Clang compiler:
    ```
    $ cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
    ```
- Enable MLPACK: 
    ```
    $ cmake -DENABLE_MLPACK=1 ..
    ```
## Contribution guidelines ##

If you would like to contribute to Edge Learning, please contact the developer:

* Mirco De Marchi <mirco.demarchi@univr.it>
* Luca Geretti <luca.geretti@univr.it>