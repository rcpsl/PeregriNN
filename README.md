# PeregriNN (CAV 21')
PeregriNN is an efficient model checker that verifies the input/output behaviour of ReLU Neural networks using search and optimization tecnhiques. It uses a unique convex objective to identify the most "problematic" neurons and use them as a heursitic to guide the search aspect. Detailed information about the solver can be found in the paper [PEREGRiNN: Penalized-Relaxation Greedy Neural Network Verifier](https://arxiv.org/abs/2006.10864).

This repository contains the implementation of PeregriNN and evalutations of adversarial robustness of Neural Networks trained on the MNIST dataset. The networks and images are the same as the ones used in [VNN 20'](https://sites.google.com/view/vnn20/vnncomp). We support `nnet` format for Neural Network files, and accept unnormalized images (check VNN directory for examples)

## System requirements

PeregriNN is tested on a fresh installation of Ubuntu 20.04.2 LTS. We recommend a single core machine with 32 GB of memory.

## Installation

We recommend using `conda` for installing PeregriNN and we'll provide a step by step guide for the setup of conda environment on a Ubuntu.

Install some packages required for the build.

`sudo apt install gcc libgmp3-dev`

Install the latest version of [Anaconda](https://docs.anaconda.com/anaconda/install/).

Create a conda environment with all the dependencies by running

`conda env create --name envname --file=requirements.yml`

make sure to replace `envname` with the environment name you'd like. Make sure the environment was created without errors.

## License 
PeregriNN relies on Gurobi commercial solver which isn't open source. However, they provide a free academic license. Please request an academic license from [here](https://www.gurobi.com/academia/academic-program-and-licenses/)
## Test installation
After installing Gurboi license, we can test the installation by running
```
conda activate envname
python -u peregriNN.py VNN/mnist-net_256X2.nnet VNN/mnist_images/image1 0.02
```
and make sure there is no error.

PeregriNN expects three positional arguments and one optional argument. The first two arguments `network` and `image` are the relative paths to the Neural network and the image files, the third arguments is the `epsilon` perturbation. The running timeout can be set by providing an optional arg `--timeout` which defaults to 300 seconds.

## Evaluation
Execute the script `./testbench` which runs a subset of the testcases of the MNIST dataset and finishes in about 1-2 hours on an off-the-shelf CPU. However, To generate the full results in the CAV21' paper, run the script `run_mnist` (This will take a long time). Executing any of those scripts will clear the `results` directory and create a new log file for each (network, epsilon) pair.
