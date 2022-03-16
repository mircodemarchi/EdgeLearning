# EdgeLearning
Deep Learning for Edge Computing

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 
[![Build Status](https://github.com/mircodemarchi/EdgeLearning/workflows/Continuous%20Integration/badge.svg)](https://github.com/mircodemarchi/EdgeLearning/actions)
[![codecov](https://codecov.io/gh/mircodemarchi/EdgeLearning/branch/main/graph/badge.svg)](https://codecov.io/gh/mircodemarchi/EdgeLearning)

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

## Performance Analysis

### MLPACK

- Regression: 
    - Increment the amount of epochs: { mean: 56.0856 sec }

| Epochs | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|--------|---|---|---|---|---|---|---|---|---|----|
| Time   | 281.002 sec | 1.07404 sec | 262.357 sec | 1.72406 sec | 1.96317 sec | 2.16733 sec | 2.37012 sec | 2.59695 sec | 2.80394 sec | 2.79867 sec |

    - Increment the training-set size: { mean: 360.687 ms }

| Training entries | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|------------------|----|----|----|----|----|----|----|----|----|-----|
| Time   | 32.82 ms | 135.138 ms | 237.426 ms | 174.33 ms | 197.046 ms | 155.427 ms | 77.231 ms | 81.525 ms | 77.702 ms | 557.085 ms | 


    - Increment the amount of layers: { mean: 35.044 sec }

| Hidden Layers | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---------------|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|----|----|
| Time   | 23.736 sec | 139.086 sec | 496.553 ms | 190.319 sec | 676.083 ms | 824.294 ms | 1.28675 sec | 1.04484 sec | 94.458 sec | 1.37685 sec | 1.50873 sec | 1.65631 sec | 122.417 sec | 107.3 sec | 2.0276 sec | 2.57166 sec | 2.33122 sec | 2.43546 sec | 2.53883 sec | 2.79024 sec |


    - Increment the shape of layers: { mean: 45.1215 sec }

| Internal layer size | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 
|---------------------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| Time   | 61.5239 sec | 232.004 sec | 448.188 ms | 454.228 ms  | 457.706 ms | 528.321 ms | 70.9858 sec | 224.465 sec | 489.811 ms | 77.5723 sec | 593.419 ms  | 585.065 ms  | 525.205 ms | 228.147 sec | 565.375 ms | 567.875 ms | 652.765 ms  | 583.536 ms  | 677.708 ms | 604.264 ms |


### EDGELEARNING

- Regression: 
    - Increment the amount of epochs: { mean: 56.0856 sec }

| Epochs | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|--------|---|---|---|---|---|---|---|---|---|----|
| Time   | 683.423 ms | 1.32282 sec | 2.06147 sec | 2.75427 sec | 3.26433 sec | 4.41129 sec | 4.66251 sec | 5.74412 sec | 6.60897 sec | 6.83326 sec |

    - Increment the training-set size: { mean: 360.687 ms }

| Training entries | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|------------------|----|----|----|----|----|----|----|----|----|-----|
| Time   | 10.49 ms | 21.256 ms | 29.558 ms | 40.76 ms | 59.008 ms | 60.613 ms | 71.152 ms | 83.192 ms | 108.185 ms | 118.731 ms | 


    - Increment the amount of layers: { mean: 35.044 sec }

| Hidden Layers | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---------------|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|----|----|
| Time   | 70.101 ms | 174.788 ms | 242.488 ms | 341.363 ms | 429.047 ms | 541.926 ms | 561.262 ms | 642.312 ms | 721.369 ms | 900.54 ms | 966.756 ms | 1.08307 sec | 1.10536 sec | 1.20248 sec | 1.24223 sec | 1.30271 sec | 1.42063 sec | 1.47547 sec | 1.54038 sec | 1.62665 sec |


    - Increment the shape of layers: { mean: 45.1215 sec }

| Internal layer size | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 
|---------------|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|----|----|
| Time   | 237.312 ms | 253.495 ms | 297.243 ms | 305.624 ms | 344.783 ms | 373.822 ms | 410.996 ms | 422.439 ms | 485.472 ms | 489.93 ms | 542.076 ms | 564.976 ms | 629.215 ms | 643.841 ms | 771.383 ms | 788.164 ms | 1.3385 sec | 979.534 ms | 951.251 ms | 943.798 ms |
