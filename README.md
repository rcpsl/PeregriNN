# PeregriNN (CAV 21')
PeregriNN is an efficient model checker that verifies the input/output behaviour of ReLU Neural networks using search and optimization tecnhiques. It uses a unique convex objective to identify the most "problematic" neurons and use them as a heursitic to guide the search aspect. Detailed information about the solver can be found in the paper [PEREGRiNN: Penalized-Relaxation Greedy Neural Network Verifier](https://arxiv.org/abs/2006.10864).

This repository contains the implementation of PeregriNN and evalutations of adversarial robustness of Neural Networks trained on the MNIST dataset. The networks and data are the same as the ones used in [VNN 20'](https://sites.google.com/view/vnn20/vnncomp) and are in the `nnet` format.

## Installation

We recommend using `conda` for installing PeregriNN and we'll provide a step by step guide for the setup of conda environment on a linux OS.

Install some packages required for the build.

`sudo apt install gcc libgmp3-dev`

Install the latest version of [Anaconda](https://docs.anaconda.com/anaconda/install/).

Create a conda environment with all the dependencies by running

`conda env create --name envname --file=requirements.yml`

make sure to replace `envname` with the environment name you'd like. Make sure the environment was created without errors.

## Test installation
```
conda activate envname
python -u peregriNN.py VNN/mnist-net_256X2.nnet VNN/mnist_images/image1 0.02
```
and make sure there is no error.

PeregriNN expects three positional arguments and one optional argument. The first two arguments `network` and `image` are the full paths to the Neural network and the image files, the third arguments is the `epsilon` perturbation. The running timeout can be set by providing an optional arg `--timeout` which defaults to 300 seconds.

## Evaluation
To generate the results in the CAV21' paper, run the script `run_mnist`. However, this will take a long time, so we provided another script `test_bench` which runs a subset of the testcases. Executing any of those scripts will clear the `results` directiory and create a new log file for each (network, eps) pair.
