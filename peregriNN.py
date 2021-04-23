from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *
from multiprocessing import Process, Value
import argparse
from os import path

eps = 1E-10

class TimeOutException(Exception):
    def __init__(self, *args, **kwargs):
        pass

def print_summary(network,prop, safety, time, extra = None):
    if(extra is not None):
        print(network[14:19],prop,safety,time,extra)
    else:
        print(network[14:19],prop,safety,time)

def alarm_handler(signum, frame):
    raise TimeOutException()

def check_property(network,x,target):
    u = network.evaluate(x)
    if(np.argmax(u) != target):
        # print("Potential CE succeeded")
        return True
    return False

def check_prop_samples(nn,samples,target):
    outs = nn.evaluate(samples.T).T
    outs = np.argmax(outs,axis = 1)
    return np.any(outs  != target)


def run_instance(network, input_bounds, check_property, target, out_idx, adv_found,convex_calls = 0, max_depth =30):

    try:
        solver = Solver(network = network,property_check=check_property,target = target,convex_calls=convex_calls,MAX_DEPTH=max_depth)
        # input_vars = [solver.state_vars[i] for i in range(len(solver.state_vars))]
        A = np.eye(network.image_size)
        lower_bound = input_bounds[:,0]
        upper_bound = input_bounds[:,1]
        solver.add_linear_constraints(A,solver.in_vars_names,lower_bound,GRB.GREATER_EQUAL)
        solver.add_linear_constraints(A,solver.in_vars_names,upper_bound,GRB.LESS_EQUAL)
        
        output_vars = [solver.out_vars[i] for i in range(len(solver.out_vars))]
        A = np.zeros(network.output_size)
        A[out_idx] = 1
        A[target] = -1
        b = [eps]
        solver.add_linear_constraints([A],solver.out_vars_names,b,GRB.GREATER_EQUAL)
        
        solver.preprocessing = False
        vars,status = solver.solve()
        if(status == 'SolFound'):
            adv_found.value = 1
        return status
        # print('Terminated')
    except Exception as e:
        raise e



def main(args):


    #Parse args
    nnet = args.network
    image_file = args.image
    img_name = image_file.split('/')[-1]
    delta = float(args.eps)
    TIMEOUT = int(args.timeout)
    MAX_DEPTH = int(args.timeout)
    #Init NN structure
    nn = NeuralNetworkStruct()
    nn.parse_network(nnet,type = 'mnist')
    # print('Loaded network:',nnet)
    
    with open(image_file,'r') as f:
        image = f.readline().split(',')
        image = np.array([float(num) for num in image[:-1]]).reshape((-1,1))/255.0
        output = nn.evaluate(image)
        target = np.argmax(output)
        nn.set_target(target)
        other_ouputs = [i for i in range(nn.output_size) if i != target]
        # print('Testing',image_file)
        # print('Output:',output,'\nTarget-->',target)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(TIMEOUT)
    try:
        start_time = time()
        # print('Norm:',delta)
        #Solve the problem for each other output
        lb = np.maximum(image-delta,0.0)
        ub = np.minimum(image+delta,1.0)
        input_bounds = np.concatenate((lb,ub),axis = 1)
        samples = sample_network(nn,input_bounds,15000)
        SAT = check_prop_samples(nn,samples,target)
        if(SAT):
        #    adv +=1
           print_summary(nnet,img_name,'unsafe',time()-start_time)
           return
        nn.set_bounds(input_bounds)
        out_list_ub = copy(nn.layers[nn.num_layers-1]['conc_ub'])
        other_ouputs = np.flip(np.argsort(out_list_ub,axis = 0)).flatten().tolist()
        other_ouputs = [idx for idx in other_ouputs if idx!= target and out_list_ub[idx] > 0]
        adv_found = Value('i',0)
        result = ''
        for out_idx in other_ouputs:
            if 0 > nn.layers[len(nn.layers)-1]['conc_ub'][out_idx]:
                continue
            #print('Testing Adversarial with label', out_idx)
            network = deepcopy(nn)
            result = run_instance(network, input_bounds, check_property, target, out_idx,adv_found,max_depth = MAX_DEPTH)
            if(result == 'SolFound'):
                break
        #signal.alarm(0)
        if(result == 'SolFound'):
            print_summary(nnet,img_name,'unsafe',time() - start_time)
        else:
            print_summary(nnet,img_name,'safe',time()-start_time)

    except TimeOutException:
        print_summary(nnet,img_name,'timeout',TIMEOUT)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PeregriNN model checker")
    parser.add_argument('network',help="path to neural network nnet file")
    parser.add_argument('image',help="path to image file")
    parser.add_argument('eps',help="eps perturbation")
    parser.add_argument('--timeout',default=300,help="timeout value")
    parser.add_argument('--max_depth',default=30,help="Maximum exploration depth")
    args = parser.parse_args()

    main(args)
