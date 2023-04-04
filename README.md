# PeregriNN (CAV 21')
PeregriNN is an efficient model checker that verifies the input/output behaviour of ReLU Neural networks using search and optimization tecnhiques. It uses a unique convex objective to identify the most "problematic" neurons and use them as a heursitic to guide the search aspect. Detailed information about the solver can be found in the paper [PEREGRiNN: Penalized-Relaxation Greedy Neural Network Verifier](https://arxiv.org/abs/2006.10864).

This repository contains the version of PeregriNN submitted to [VNN 2022 competition](https://sites.google.com/view/vnn2022). This version supports networks in `ONNX` format, and accepts specifications in `VNNLIB` format.

## System requirements

PeregriNN is tested on a fresh installation of Ubuntu 20.04.2 LTS. We recommend a single core machine with 32 GB of memory.

## Installation

We recommend using `conda` for installing PeregriNN and we'll provide a step by step guide for the setup of conda environment on a Ubuntu.

Install some packages required for the build.

`sudo apt install gcc libgmp3-dev`

Install the latest version of [Anaconda](https://docs.anaconda.com/anaconda/install/).

Create a conda environment with all the dependencies by running

`conda env create --name envname --file=environment.yml`

make sure to replace `envname` with the environment name you'd like. Make sure the environment was created without errors.

## License 
PeregriNN relies on Gurobi commercial solver which isn't open source. However, they provide a free academic license. Please request an academic license from [here](https://www.gurobi.com/academia/academic-program-and-licenses/)
## Test installation
After installing Gurboi license, we can test the installation by running
```
conda activate envname
python -u peregriNN.py examples/mnistfc/mnist-net_256x2.onnx examples/mnistfc/prop_2_0.03.vnnlib --category mnist_fc
```
and make sure the program terminates without error, you can check `out.txt` to see the verification result.

## VNN 2022 competition benchmarks
To evaluate PeregriNN on VNN 2022 benchmarks, run `git submodule update --init --recursive` to pull the benchmark repo `vnncomp2022_benchmarks`. Extract all the test cases for the benchmarks by running 
```
cd vnncomp2022_benchmarks
./setup.sh
```
You can run PeregriNN on the benchmarks using `run_all_categories.sh` and providing the path to PeregriNN's `vnn_scripts` folder as well as the names of the desired benchmarks. Please check `run_all_categories.sh` for the arguments list.

## Running PeregriNN on a specific instance

To run PeregriNN on a single instance of the supported datasets, run the following command

`python -u peregriNN.py $model $spec --category $CATEGORY --result_file $FILE`

where the first two positional arguments `model` and `spec` are the relative paths to the Neural network ONNX file and the VNNLIB spec file, the optional arguments include `--timeout` (defaults to 300 seconds) and `--category` which determines which dataset you are working with (default is mnisft_fc). The `--result_file` argument specifies the verification result output file (defaults to `out.txt`). Please check `utils/dataset_info.py` and `utils/config.py` to know the supported datasets and to add other datasets.

