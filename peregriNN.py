import sys,os

# os.environ['MKL_NUM_THREADS']="1"
# os.environ['NUMEXPR_NUM_THREADS']="1"
# os.environ['OMP_NUM_THREADS']="2"
# os.environ['OPENBLAS_NUM_THREADS']="1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from peregrinn.verifier import ResultType, Verifier
from utils.config import Setting
from utils.specification import Specification
from parsers.onnx_parser import ONNX_Parser
from solver import *
import time
from random import random, seed
import numpy as np
import signal
import glob
from multiprocessing import Value
from NeuralNetwork import *
# import multiprocessing as mp
# from multiprocessing import Process, Value
import argparse
from os import path
import logging
from parsers.vnnlib import VNNLib_parser
import torch
from utils.datasets_info import Dataset_MetaData

#TODO: DO BETTER :D  (find a way that is more flexible than just importing those operators)
from operators.linear import * 
from operators.flatten import *
from operators.activations import *

import warnings
from utils.Logger import get_logger
logger = get_logger('main')
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


def old(args):


    #Parse args
    nnet = args.model
    image_file = args.image
    img_name = image_file.split('/')[-1]
    delta = float(args.eps)
    TIMEOUT = int(args.timeout)
    MAX_DEPTH = int(args.timeout)
    #Init NN structure
    onnx_parser = ONNX_Parser(nnet)
    torch_model = onnx_parser.to_pytorch()
    nn = PytorchNN()
    nn.parse_network(torch_model, 784)
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
        start_time = time.time()
        # print('Norm:',delta)
        #Solve the problem for each other output
        lb = np.maximum(image-delta,0.0)
        ub = np.minimum(image+delta,1.0)
        input_bounds = np.concatenate((lb,ub),axis = 1)
        # samples = sample_network(nn,input_bounds,15000)
        # SAT = check_prop_samples(nn,samples,target)
        # if(SAT):
        # #    adv +=1
        #    print_summary(nnet,img_name,'unsafe',time.time()-start_time)
        #    return
        nn.set_bounds(input_bounds)
        # out_list_ub = copy(nn.layers[nn.num_layers-1]['conc_ub'])
        # other_ouputs = np.flip(np.argsort(out_list_ub,axis = 0)).flatten().tolist()
        # other_ouputs = [idx for idx in other_ouputs if idx!= target and out_list_ub[idx] > 0]
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
            print_summary(nnet,img_name,'unsafe',time.time() - start_time)
        else:
            print_summary(nnet,img_name,'safe',time.time()-start_time)

    except TimeOutException:
        print_summary(nnet,img_name,'timeout',TIMEOUT)

def main(args):

    
    tic = time.perf_counter()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(args.timeout)
    try:
        model_path = args.model
        vnnlib_filename = args.spec
        dataset = args.dataset
        onnx_parser = ONNX_Parser(model_path, simplify = args.simplify)
        vnnlib_parser = VNNLib_parser(dataset = dataset)
        in_shape = Dataset_MetaData.inout_shapes[dataset]['input']
        out_shape = Dataset_MetaData.inout_shapes[dataset]['output']
        vnnlib_spec = vnnlib_parser.read_vnnlib_simple(vnnlib_filename, in_shape.prod().item(),
                                                        out_shape.prod().item())
        # device = torch.device('cpu')
        torch_model = onnx_parser.to_pytorch()
        if(Setting.TORCH_PRECISION == torch.float64):
            torch_model = torch_model.double()
        torch_model.eval()
        for name, param in torch_model.named_parameters():
            param.requires_grad  = False

        if(args.subtract_target):
            nom_input = vnnlib_spec.input_bounds.mean(dim = 1)
            target = torch_model(nom_input.reshape(*in_shape).unsqueeze(0)).argmax()
            out_layer = list(torch_model.modules())[-1]
            out_layer.weight -= out_layer.weight[target].clone()
            out_layer.bias -= out_layer.bias[target].clone()
            
        unsafe_objectives = [i for i in range(len(vnnlib_spec.objectives))]
        verifier = Verifier(torch_model, vnnlib_spec)
        if(Setting.TRY_SAMPLING):
            result = verifier.check_by_sampling()
            if(result.status == ResultType.SOL_FOUND):
                #TODO: write result to file
                logger.info('Solution found by sampling')

        if(Setting.TRY_OVERAPPROX and verifier.verification_result.status != ResultType.SOL_FOUND):
            status, unsafe_objectives = verifier.quick_check_bounds()
            if(status == ResultType.NO_SOL):
                #TODO: write result to file
                logger.debug('Property Safe by overapproximation')

        if(verifier.verification_result.status == ResultType.UNKNOWN):
            logger.info("Formally verify property")
            for i in unsafe_objectives:
                logger.debug(f"Verifying property {i}")
                verifier.verify(objective_idx= i )
                if verifier.verification_result.status == ResultType.SOL_FOUND:
                    logger.debug(f"Verifier found a Counterexample : Objective {i}")
                    break

        #TODO: write result to file
        total_time = time.perf_counter() - tic
        if(verifier.verification_result.status != ResultType.SOL_FOUND):
            logger.info(f"Property proved safe")
            # with open('test.txt','a') as f:
            #     f.write(f'{delta},{img_name},safe,{total_time:.3f}\n')

        else:
            # with open('test.txt','a') as f:
            #     f.write(f'{delta},{img_name},unsafe,{total_time:.3f}\n')
            logger.info(f"Property violated !")
    except TimeOutException as e:
        #Cleanup
        verifier.cleanup()
        # with open('test.txt','a') as f:
        #     f.write(f'{delta},{img_name},timeout,{time.perf_counter() - tic:.3f}\n')
        logger.info("Timoeut")
    except Exception as e:
        logger.exception(str(e))
        raise e
    logger.info(f"Total program time: {time.perf_counter() - tic:.2f}")

def test_verifier(args):
    tic = time.perf_counter()
    model_path = args.model
    vnnlib_filename = args.spec
    dataset = args.dataset
    onnx_parser = ONNX_Parser(model_path)
    vnnlib_parser = VNNLib_parser(dataset = dataset)
    in_shape = Dataset_MetaData.inout_shapes[dataset]['input']
    out_shape = Dataset_MetaData.inout_shapes[dataset]['output']
    vnnlib_spec = vnnlib_parser.read_vnnlib_simple(vnnlib_filename, in_shape.prod().item(),
                                                    out_shape.prod().item())
    # device = torch.device('cpu')
    torch_model = onnx_parser.to_pytorch()
    if(Setting.TORCH_PRECISION == torch.float64):
        torch_model = torch_model.double()
    torch_model.eval()
    for _, param in torch_model.named_parameters():
        param.requires_grad  = False

    verifier = Verifier(torch_model, vnnlib_spec, i)
    for i in range(len(vnnlib_spec.objectives)):
        verifier.verify()
        if verifier.verification_result.status == ResultType.SOL_FOUND:
            break
    logger.info(f"Total program time: {time.perf_counter() - tic:.2f}")

def mnist_verify(args):
    tic = time.perf_counter()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(args.timeout)
    try:
        model_path = args.model
        # vnnlib_filename = args.spec
        dataset = args.dataset
        image_file = args.image
        img_name = image_file.split('/')[-1]
        delta = float(args.eps)
        onnx_parser = ONNX_Parser(model_path)
        vnnlib_parser = VNNLib_parser(dataset = dataset)
        in_shape = Dataset_MetaData.inout_shapes[dataset]['input']
        out_shape = Dataset_MetaData.inout_shapes[dataset]['output']
        # vnnlib_spec = vnnlib_parser.read_vnnlib_simple(vnnlib_filename, in_shape.prod().item(),
        #                                                 out_shape.prod().item())
        torch_model = onnx_parser.to_pytorch()
        if(Setting.TORCH_PRECISION == torch.float64):
            torch_model = torch_model.double()
        torch_model.eval()
        for _, param in torch_model.named_parameters():
            param.requires_grad  = False

        with open(image_file,'r') as f:
            image = f.readline().split(',')
            image = torch.tensor([float(num) for num in image[:-1]], dtype = Setting.TORCH_PRECISION)/255.0
            target = torch_model(image.unsqueeze(0)).argmax()
        lb = torch.maximum(torch.zeros_like(image), image - delta)
        ub = torch.minimum(torch.ones_like(image), image + delta)
        input_bounds = torch.stack((lb,ub), dim = -1)
        objectives = []
        for i in range(10):
            if i != target:
                A = np.zeros((1,10))
                A[0,target] = 1
                A[0,i] = -1
                b = np.array([0])
                objectives.append((A,b))
        out_layer = list(torch_model.modules())[-1]
        out_layer.weight -= out_layer.weight[target].clone()
        out_layer.bias -= out_layer.bias[target].clone()
        spec = [(input_bounds, objectives)]
        vnnlib_spec = Specification(spec,dataset = 'mnistfc')
        

        verifier = Verifier(torch_model, vnnlib_spec)
        unsafe_objectives = [i for i in range(len(vnnlib_spec.objectives))]

        if(Setting.TRY_SAMPLING):
            result = verifier.check_by_sampling()
            if(result.status == ResultType.SOL_FOUND):
                #TODO: write result to file
                logger.info('Solution found by sampling')

        if(Setting.TRY_OVERAPPROX and verifier.verification_result.status != ResultType.SOL_FOUND):
            status, unsafe_objectives = verifier.quick_check_bounds()

            if(status == ResultType.NO_SOL):
                #TODO: write result to file
                logger.debug('Property Safe by overapproximation')

        if(verifier.verification_result.status == ResultType.UNKNOWN):
            logger.info("Formally verify property")
            for i in unsafe_objectives:
                logger.debug(f"Verifying property {i}")
                verifier.verify(objective_idx= i )
                if verifier.verification_result.status == ResultType.SOL_FOUND:
                    logger.debug(f"Verifier found a Counterexample : Objective {i}")
                    break

        #TODO: write result to file
        total_time = time.perf_counter() - tic
        if(verifier.verification_result.status != ResultType.SOL_FOUND):
            logger.info(f"Property proved safe")
            with open('test.txt','a') as f:
                f.write(f'{delta},{img_name},safe,{total_time:.3f}\n')

        else:
            with open('test.txt','a') as f:
                f.write(f'{delta},{img_name},unsafe,{total_time:.3f}\n')
            logger.info(f"Property violated !")
    except TimeOutException as e:
        #Cleanup
        verifier.cleanup()
        with open('test.txt','a') as f:
            f.write(f'{delta},{img_name},timeout,{time.perf_counter() - tic:.3f}\n')
            logger.info("Timoeut")
        pass
    logger.info(f"Total program time: {time.perf_counter() - tic:.2f}")
        
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="PeregriNN model checker")
    parser.add_argument('model',help="path to neural network ONNX file")
    parser.add_argument('spec',help="path to vnnlib specification file")
    parser.add_argument('--image',type = str,help="Maximum perturbation")
    parser.add_argument('--dataset', type = str, default = 'mnistfc')
    parser.add_argument('--timeout',type = int, default=300,help="timeout value")
    parser.add_argument('--max_depth',default=30,help="Maximum exploration depth")
    parser.add_argument('--eps',default=0.02,help="Maximum perturbation")
    parser.add_argument('--simplify', default = False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--subtract_target', default = False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    #Root logger config
    log_dir = os.path.join(os.getcwd(),'logs')
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir)
    # log_file = os.path.join(log_dir,'logs.log')
    # img_name = args.image.split('/')[-1]
    model_name = args.model.split('/')[-1].split('.')[0]

    # log_file = os.path.join(log_dir,f"{model_name}_{img_name}_{args.eps}.log")
    log_file = os.path.join(log_dir,f"logs.log")

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        level = Setting.LOG_LEVEL,
        handlers=[
            logging.FileHandler(log_file, mode ='w')
        ]
    )
    logging.captureWarnings(True)

    
    torch.multiprocessing.set_start_method("fork")
    # test_verifier(args)
    # old(args)
    # mnist_verify(args)
    main(args)
