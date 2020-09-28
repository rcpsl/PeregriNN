import numpy as np
from random import random,seed, uniform,randint
from NeuralNetwork import *
from time import time
from copy import deepcopy,copy

def print_summary(network,prop, safety, time):
    print('%s,%d,%s,%f'%(network[20:23],prop,safety,time))
    
def sample_network(nn,bounds,num_samples):
    start  = time()
    lower_bounds, upper_bounds = bounds[:,0],bounds[:,1]
    np.random.seed(5)
    samples = np.random.uniform(lower_bounds, upper_bounds,size = (num_samples,len(lower_bounds)))
    #
    # phases,y = nn.get_phases(samples)
    # phases = phases[:-1]
    # count_regions(nn,phases)
    return samples

def count_regions(nn, phases):
    
    all_phases = np.zeros((phases[0].shape[0],0))
    regions = {}
    fixed_relus = [neuron_idx for layer,neuron_idx in nn.active_relus + nn.inactive_relus if layer ==1] 
    top_k = list(set(range(nn.layers_sizes[1])) - set(fixed_relus))
    for layer_phase in phases:
        all_phases = np.hstack((all_phases,layer_phase)).astype(int)
    for sample_phase in all_phases:
        fl = int(np.array2string(sample_phase[top_k],max_line_width = 10000000, separator = '')[1:-1],2)
        hash_val = hash(fl)
        if(hash_val in regions):
            regions[hash_val] += 1
        else:
            regions[hash_val] = 1
            # regions[has]

    return all_phases


def count_changes_layer(nn,layer_idx,phases,fixed_relus):
   
    layer_counts = []
    neurons_counts = []
    for neuron_idx in range(nn.layers_sizes[layer_idx]):
        active_samples = np.where(phases[layer_idx-1][:,neuron_idx]  == True)[0]
        inactive_samples = np.where(phases[layer_idx-1][:,neuron_idx]  == False)[0]
        for fixed_neuron in fixed_relus:
            l_idx,n_idx,phase = fixed_neuron
            active_samples = np.where(phases[l_idx-1][active_samples,n_idx]  == phase)[0]
            inactive_samples = np.where(phases[l_idx-1][inactive_samples,n_idx]  == phase)[0]
        active_changes = 0
        inactive_changes = 0
        for layer_phases in phases[layer_idx-1:]:
            if(active_samples.shape[0] > 0):
                active_count = np.sum(layer_phases[active_samples],axis = 0)
                active_changes += np.sum(np.logical_and(active_count > 0 , active_count < active_samples.shape[0]))
            if(inactive_samples.shape[0] > 0):
                inactive_count = np.sum(layer_phases[inactive_samples],axis = 0)
                inactive_changes += np.sum(np.logical_and(inactive_count > 0 , inactive_count < inactive_samples.shape[0]))

        neurons_counts.append((active_changes,inactive_changes))
    return np.array(neurons_counts)
    pass

def count_changes(nn,phases):
   
    layer_counts = []
    for layer_idx in range(1,nn.num_layers-1):
        neurons_counts = []
        for neuron_idx in range(nn.layers_sizes[layer_idx]):
            active_samples = np.where(phases[layer_idx-1][:,neuron_idx]  == True)[0]
            inactive_samples = np.where(phases[layer_idx-1][:,neuron_idx]  == False)[0]
            active_changes = 0
            inactive_changes = 0
            for layer_phases in phases[layer_idx-1:]:
                if(active_samples.shape[0] > 0):
                    active_count = np.sum(layer_phases[active_samples],axis = 0)
                    active_changes += np.sum(np.logical_and(active_count > 0 , active_count < active_samples.shape[0]))
                if(inactive_samples.shape[0] > 0):
                    inactive_count = np.sum(layer_phases[inactive_samples],axis = 0)
                    inactive_changes += np.sum(np.logical_and(inactive_count > 0 , inactive_count < inactive_samples.shape[0]))

            neurons_counts.append((active_changes,inactive_changes))
        layer_counts.append(neurons_counts)
    return np.array(layer_counts)
    pass

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
        if(f_relus1 > max_stable or f_relus2 > max_stable):
            max_stable = max(f_relus1,f_relus2)
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
    
    
