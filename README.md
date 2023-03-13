# Implementation of Graph Self-Explaining Neural Networks (Graph-SENN)
In our method, we apply the fundamental idea of [Self-Explaining Neural Networks](https://arxiv.org/abs/1806.07538) to pooling in Graph Neural Networks.
For details, please see the accompanying project report.

## Setup
Please use
```
conda env create -f environment.yml
conda activate graph-senn
```
to set up the required packages. Depending on your local installation (CUDA version, pytorch, etc.) this might not work.
In this case it could be simplest to install the desired versions of `pytorch` and `pytorch-geometric` and progress by 
trying to start a run as shown below, installing all packages via `pip` until no error is thrown.

## Getting started

For getting started, see:
```
python train.py --help
```

### Weights & Biases
We use [Weights & Biases](https://wandb.com) for logging. This can be disabled with `--no_wandb`. However, note that you will not see any 
training or test metrics in this case. Model checkpoints can still be saved. To use weights and biases, install it via 
`pip install wandb`, login via `wandb login` and change the `jonas-juerss` in  `custom_logger.py` to your username.

### Loading and visualizing trained models
You can load and visualize trained models as shown in `visualizations.ipynb`. The easiest way to do this is to give the 
id of a wandb run, which will automatically restore the run configuration from Weights & Biases and find the correct 
local file to restore the model parameters (assuming the model was trained on the same device or the checkpoint was 
copied to the same local path).
