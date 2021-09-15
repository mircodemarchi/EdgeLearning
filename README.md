# AriadneDL
Deep Learning for Ariadne

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 
[![Build Status](https://github.com/mircodemarchi/AriadneDL/workflows/Continuous%20Integration/badge.svg)](https://github.com/mircodemarchi/AriadneDL/actions)
[![codecov](https://codecov.io/gh/mircodemarchi/AriadneDL/branch/main/graph/badge.svg)](https://codecov.io/gh/mircodemarchi/AriadneDL)

The project library is supported on MacOS and Linux systems, with Clang and GNU GCC compiler. 

N.B. The option ENABLE_MLPACK at true is not supported in MacOS with GNU GCC compiler. 

## Configuration 

- With GNU GCC compiler: 
    ```
    cmake -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 ..
    ```
- With Clang compiler:
    ```
    cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
    ```
- Enable MLPACK: 
    ```
    cmake -DENABLE_MLPACK=1 ..
    ```