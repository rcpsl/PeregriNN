#!/bin/bash

fname="mnist_random_slack.txt"
# python -u vnn.py VNN/mnist-net_256x2.nnet 0.01 >> $fname
python -u vnn.py VNN/mnist-net_256x2.nnet 0.02 >  $fname 
python -u vnn.py VNN/mnist-net_256x4.nnet 0.02 >> $fname 
python -u vnn.py VNN/mnist-net_256x6.nnet 0.02 >> $fname 
python -u vnn.py VNN/mnist-net_256x2.nnet 0.05 >> $fname 

# python -u vnn.py VNN/mnist-net_256x4.nnet 0.01 >> mnist_results.txt
#python -u vnn.py VNN/mnist-net_256x4.nnet 0.05 >> mnist_results.txt

# python -u vnn.py VNN/mnist-net_256x6.nnet 0.01 >> mnist_results.txt
#python -u vnn.py VNN/mnist-net_256x6.nnet 0.05 >> mnist_results.txt
