import sys,os
from turtle import update

# os.environ['MKL_NUM_THREADS']="1"
# os.environ['NUMEXPR_NUM_THREADS']="1"
# os.environ['OMP_NUM_THREADS']="2"
# os.environ['OPENBLAS_NUM_THREADS']="1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from peregrinn.verifier import ResultType, Verifier
from utils.config import Setting, update_cfg
from utils.specification import Specification
from parsers.onnx_parser import ONNX_Parser
import time
from random import random, seed
import numpy as np
import signal
import glob
from multiprocessing import Value
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

def alarm_handler(signum, frame):
    raise TimeOutException()

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
        try:
            in_shape = Dataset_MetaData.inout_shapes[dataset]['input']
            out_shape = Dataset_MetaData.inout_shapes[dataset]['output']
        except Exception as e:
            in_shape, out_shape = vnnlib_parser.get_num_inputs_outputs(model_path)
            in_shape = torch.atleast_1d(torch.tensor(in_shape,dtype = torch.int))
            out_shape = torch.atleast_1d(torch.tensor(out_shape, dtype = torch.int))
            Dataset_MetaData.inout_shapes[dataset] = {'input': in_shape, 'output': out_shape}
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
                    logger.info(f"Verifier found a Counterexample : Objective {i}")
                    break

        #TODO: write result to file
        total_time = time.perf_counter() - tic
        if(verifier.verification_result.status != ResultType.SOL_FOUND):
            logger.info(f"Property proved safe")
            with open(args.result_file,'w') as f:
                f.write('unsat\n')

        else:
            logger.info(f"Property violated !")
            vnnlib_result = vnnlib_parser.result_str(*verifier.verification_result.ce.values())
            with open(args.result_file,'w') as f:
                f.write('sat\n')
                f.write(vnnlib_result)
            logger.info('Counter example:\n' + vnnlib_result)
    except TimeOutException as e:
        #Cleanup
        verifier.cleanup()
        with open(args.result_file,'w') as f:
            f.write('timeout\n')
        logger.info("Timoeut")
    except Exception as e:
        logger.exception(str(e))
        raise e
    logger.info(f"Total program time: {time.perf_counter() - tic:.2f}")
        
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="PeregriNN model checker")
    parser.add_argument('model',help="path to neural network ONNX file")
    parser.add_argument('spec',help="path to vnnlib specification file")
    # parser.add_argument('--image',type = str,help="Maximum perturbation")
    parser.add_argument('--dataset', type = str, default = 'mnistfc')
    parser.add_argument('--timeout',type = int, default=300,help="timeout value")
    parser.add_argument('--max_depth',default=30,help="Maximum exploration depth")
    # parser.add_argument('--eps',default=0.02,help="Maximum perturbation")
    parser.add_argument('--simplify', default = False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--subtract_target', default = False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--category', type = str, default='mnisft_fc')
    parser.add_argument('--result_file', type =str, default='out.txt')
    args = parser.parse_args()
    #Root logger config
    update_cfg(args)
    log_dir = os.path.dirname(__file__)
    log_dir = os.path.join(log_dir,'logs')
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir)
    model_name = args.model.split('/')[-1].split('.')[0]
    spec_name  = args.spec.split('/')[-1].split('.')[0]
    log_fname = f"log_{model_name}_{spec_name}.log"
    log_file = os.path.join(log_dir,log_fname)
    print(f'Logging to {log_file}...')
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        level = Setting.LOG_LEVEL,
        handlers=[
            logging.FileHandler(log_file, mode ='w')
        ]
    )
    logging.captureWarnings(True)

    
    torch.multiprocessing.set_start_method("fork")
    main(args)
