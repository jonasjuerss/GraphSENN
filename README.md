# Implementation of Graph Self-Explaining Neural Networks (Graph-SENN)
In our method, we apply the fundamental idea of [Self-Explaining Neural Networks](https://arxiv.org/abs/1806.07538) to pooling in Graph Neural Networks.
For details, please see the accompanying project report.

## Setup
Please use
```
conda env create -f environment.yml
conda activate graph-senn
```
to setup the required packages. Depending on your local installation (CUDA version, pytorch, etc.) this might not work.
In this case it could be simplest to install the desired versions of `pytorch` and `pytorch-geometric` and progress by 
trying to start a run as shown below, installing all packages via `pip` until no error is thrown.

## Getting started

For getting started, see:
```
python train.py --help
```