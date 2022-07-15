import numpy as np
from random import random,seed, uniform,randint
import time
from copy import deepcopy,copy
import torch
from utils.Logger import get_logger

logger = get_logger(__name__)

def print_summary(network,prop, safety, time):
    print('%s,%d,%s,%f'%(network[20:23],prop,safety,time))
    
def sample_network(bounds : torch.Tensor ,num_samples: int) -> torch.Tensor:
    start = time.perf_counter()
    lower_bounds, upper_bounds = bounds[...,0],bounds[...,1]
    uniform_samples = torch.FloatTensor(num_samples, *(lower_bounds.shape)).uniform_()
    samples = (upper_bounds-lower_bounds) * uniform_samples + lower_bounds
    end = time.perf_counter()
    logger.debug(f"Sampling time: {end-start:.2f} seconds")
    # samples = np.random.uniform(lower_bounds, upper_bounds,size = (num_samples,len(lower_bounds)))
    return samples