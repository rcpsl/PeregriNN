from solver import *
from time import time,sleep
from random import random, seed, uniform
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *
from utils.sample_network import * 
from poset import *

if __name__ == "__main__":
    network = "models/ACASXU_run2a_2_5_batch_2000.nnet"
    nnet = NeuralNetworkStruct()
    nnet.parse_network(network)
    raw_lower_bounds = np.array([55947.691, -3.141592, -3.141592, 1145, 0]).reshape((-1,1))
    raw_upper_bounds = np.array([62000, 3.141592, 3.141592, 1200, 60]).reshape((-1,1))
    lower_bounds = nnet.normalize_input(raw_lower_bounds)
    upper_bounds = nnet.normalize_input(raw_upper_bounds)
    input_bounds = np.concatenate((lower_bounds,upper_bounds),axis = 1)
    




    W = nnet.layers[1]['weights']
    b = nnet.layers[1]['bias']
    problems = split_input_space(nnet,input_bounds,128)
    for input_bounds in problems:
        dims=5
        s = time()
        poset = Poset(input_bounds,W[:dims],b[:dims])
        poset.build_poset()
        compute_successors(poset.root)
        print(time()-s,len(poset.hashMap))
    pass