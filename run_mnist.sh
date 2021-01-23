#!/bin/bash

# python -u vnn.py VNN/mnist-net_256x2.nnet 0.01 >> mnist_results.txt
# python -u vnn.py VNN/mnist-net_256x2.nnet 0.02 > mnist_results.txt
python -u vnn.py VNN/mnist-net_256x4.nnet 0.02 >> mnist_results.txt
python -u vnn.py VNN/mnist-net_256x6.nnet 0.02 >> mnist_results.txt
python -u vnn.py VNN/mnist-net_256x2.nnet 0.05 >> mnist_results.txt

# python -u vnn.py VNN/mnist-net_256x4.nnet 0.01 >> mnist_results.txt
python -u vnn.py VNN/mnist-net_256x4.nnet 0.05 >> mnist_results.txt

# python -u vnn.py VNN/mnist-net_256x6.nnet 0.01 >> mnist_results.txt
python -u vnn.py VNN/mnist-net_256x6.nnet 0.05 >> mnist_results.txt
