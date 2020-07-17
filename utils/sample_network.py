import numpy as np
from random import random,seed, uniform,randint
from NeuralNetwork import *
from time import time
from copy import deepcopy,copy

def print_summary(network,prop, safety, time):
    print('%s,%d,%s,%f'%(network[20:23],prop,safety,time))
    
def sample_network(nn,bounds):
    start  = time()
    lower_bounds, upper_bounds = bounds[:,0],bounds[:,1]
    err_bound = (upper_bounds - lower_bounds)/10.
    num_samples = 25
    global_err = []
    for _ in range(num_samples):
        sample = uniform(lower_bounds, upper_bounds)
        activations = nn.evaluate(sample)
        activations[activations>0] = 1
        activations[activations<=0] = 0
        err = []
        n_samples = 5
        for _ in range(n_samples):
            new_sample = uniform(sample-err_bound, sample + err_bound)
            new_act = nn.evaluate(new_sample)
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

def split_input_space(nn,bounds,MAX_SPLITS = 128):
    splits = []
    splits.append(bounds)
    while(len(splits) < MAX_SPLITS):
        # splits = sorted(splits,key=lambda x: (x[0]))
        new_splits = []
        for interval_bound in splits:
            dim_to_split = np.argmax(interval_bound[:,1] - interval_bound[:,0])
            int1,int2 = split_interval(interval_bound,dim_to_split)
            disp1 = disp2 = 1
            new_splits.append(int1)
            new_splits.append(int2)
        splits = new_splits
    return splits

def split_input_space2(nn,bounds,MAX_SPLITS = 128):
    splits = []
    disparity = []
    DISP_THRESH  = 0.2
    splits.append(bounds)
    disparity.append(sample_network(nn,bounds))
    change = True
    while(len(splits) < MAX_SPLITS and change):
        # splits = sorted(splits,key=lambda x: (x[0]))
        new_splits = []
        change = False
        delete_me = []
        for idx,interval_bound in enumerate(copy(splits)):
            if disparity[idx] > DISP_THRESH:
                dim_to_split = np.argmax(interval_bound[:,1] - interval_bound[:,0])
                int1,int2 = split_interval(interval_bound,dim_to_split)
                disp1 = sample_network(nn,int1)
                disp2 = sample_network(nn,int2)
                new_splits.append(int1)
                new_splits.append(int2)
                disparity.append(disp1)
                disparity.append(disp2)
                delete_me.append(idx)
                change = True
        splits +=new_splits
        for idx in delete_me:
            del splits[idx]
            del disparity[idx]
    return splits
def pick_dim(nn,bounds):

    max_stable = 0
    split_dim = -1
    interval1 = None
    interval2 = None
    for dim in range(nn.image_size):
        int1,int2 = split_interval(bounds,dim)
        nn1 = deepcopy(nn)
        nn2 = deepcopy(nn)
        nn1.set_bounds(int1)
        nn2.set_bounds(int2)
        f_relus1 = len(nn1.active_relus) + len(nn1.inactive_relus)
        f_relus2 = len(nn2.active_relus) + len(nn2.inactive_relus)
        avg = f_relus1 + f_relus2
        # if(f_relus1 > max_stable or f_relus2 > max_stable):
        if avg > max_stable:
            max_stable = avg
            split_dim = dim
            interval1,interval2 = int1,int2


    return split_dim,interval1,interval2

def split_input_space1(nn,bounds,MAX_SPLITS = 100):
    splits = []
    seed(5)
    splits.append(bounds)
    while(len(splits) < MAX_SPLITS):
        new_splits = []
        for interval_bound in splits:
        # interval_bound = splits.pop(-1)
            dim_to_split,int1,int2 = pick_dim(nn,interval_bound)
            new_splits.append(int1)
            new_splits.append(int2)
        splits = new_splits

    
    return splits
    
    
