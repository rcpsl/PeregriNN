import sys,os
# os.environ['MKL_NUM_THREADS']="1"
# os.environ['NUMEXPR_NUM_THREADS']="1"
os.environ['OMP_NUM_THREADS']="1"
# os.environ['OPENBLAS_NUM_THREADS']="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from parsers.onnx_parser import ONNX_Parser
from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
import glob
from NeuralNetwork import *
from multiprocessing import Process, Value
import argparse
from os import path
import logging
from utils.config import Settings 
from utils.vnnlib import read_vnnlib_simple
from intervals.symbolic_interval import SymbolicInterval
from intervals.interval_network import IntervalNetwork
import torch

#TODO: DO BETTER :D  (find a way that is more flexible than just importing those operators)
from operators.linear import * 
from operators.flatten import *
from operators.activations import *
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
    op_dict ={"Flatten":Flatten, "ReLU": ReLU, "Linear": Linear, "Conv2d": Conv2d }
    model_path = args.model
    vnnlib_filename = args.spec

    onnx_parser = ONNX_Parser(model_path)

    vnnlib_spec = read_vnnlib_simple(vnnlib_filename, 3072, 10)
    device = torch.device('cuda')
    input_bounds = torch.tensor(vnnlib_spec[0][0],dtype = torch.float32)
    torch_model = onnx_parser.to_pytorch()
    torch_model.eval()
    for name, param in torch_model.named_parameters():
        param.requires_grad  = False
    s = time()
    torch_model = torch_model.to(device)
    print(f"Time to move the model to {device}", time()-s)
    int_net = IntervalNetwork(torch_model, input_bounds, operators_dict=op_dict, in_shape = (3,32,32))
    n= input_bounds.shape[0]
    I = np.zeros((n, n+ 1), dtype = np.float32)
    np.fill_diagonal(I,1)
    I = torch.tensor(I).unsqueeze(0)
    # I = I.repeat(50,1,1)
    input_bounds = input_bounds.unsqueeze(0).unsqueeze(1)
    layer_sym = SymbolicInterval(input_bounds.to(device),I,I, device = device)
    layer_sym.concretize()
    steps = 1
    s = time()
    for iter in range(steps):
        int_net(layer_sym)
    e = time()
    print("Sym analysis with torch:", (e-s)/ steps)
    sys.exit()
    nnet = PytorchNN()
    nnet.parse_network(torch_model, 784)
    s = time()
    for iter in range(steps):
        nnet.set_bounds(input_bounds)
    e = time()
    print("Old Sym analysis with np:", (e-s)/ steps)

    # print(nnet.layers[-1]['conc_lb'])
    # print(int_net.layers[-1].post_symbolic.conc_lb)
    
    pass    
    
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
    parser.add_argument('model',help="path to neural network ONNX file")
    parser.add_argument('spec',help="path to vnnlib specification file")
    parser.add_argument('--timeout',default=300,help="timeout value")
    parser.add_argument('--max_depth',default=30,help="Maximum exploration depth")
    args = parser.parse_args()


    main(args)
