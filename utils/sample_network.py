import numpy as np
from random import random,seed, uniform,randint
from NeuralNetwork import *
from time import time
def sample_network(nn,bounds):
    start  = time()
    lower_bounds, upper_bounds = bounds[:,0],bounds[:,1]
    err_bound = (upper_bounds - lower_bounds)/10.
    num_samples = 3
    global_err = []
    for _ in range(num_samples):
        sample = uniform(lower_bounds, upper_bounds)
        _, activations = nn.evaluate(sample)
        activations[activations>0] = 1
        activations[activations<=0] = 0
        err = []
        n_samples = 5
        for _ in range(n_samples):
            new_sample = uniform(sample-err_bound, sample + err_bound)
            _, new_act = nn.evaluate(new_sample)
            new_act[new_act>0] = 1
            new_act[new_act<=0] = 0
            err.append(np.sum(new_act != activations))
        global_err.append(sum(err))
    disparity = np.mean(global_err)
    # print(time()-start)
    return disparity


def split_interval(interval_bound,dim_to_split):
    int1 = np.copy(interval_bound)
    int2 = np.copy(interval_bound)
    mid = (interval_bound[dim_to_split,0] + interval_bound[dim_to_split,1]) /2.0 
    int1[dim_to_split,1] = int2[dim_to_split,0] = mid
    return int1,int2

def split_input_space(nn,bounds,MAX_SPLITS = 100):
    DISP_THRESH  = 30.0
    splits = []
    # seed(5)
    max_disparity = sample_network(nn,bounds)
    splits.append((max_disparity,bounds))
    while(max_disparity > DISP_THRESH and len(splits) <= MAX_SPLITS):
        splits = sorted(splits,key=lambda x: (x[0]))
        _,interval_bound = splits.pop(-1)
        # dim_to_split = randint(0,nn.image_size-1)
        dim_to_split = np.argmax(interval_bound[:,1] - interval_bound[:,0])
        int1,int2 = split_interval(interval_bound,dim_to_split)
        disp1 = sample_network(nn,int1)
        disp2 = sample_network(nn,int2)
        max_disparity = max(disp1,disp2)
        splits.append((disp1,int1))
        splits.append((disp2,int2))

    
    return sorted(splits,key=lambda x: (x[0]))
    
    
