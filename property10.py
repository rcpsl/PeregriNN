from solver import *
from time import time,sleep
from random import random, seed, uniform
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *
from utils.sample_network import * 
from multiprocessing import Process,Value
eps = 1E-3
#  [Clear-of-Conflict, weak left, weak right, strong left, strong right]
class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    print('TIMEOUT!')
    raise TimeOutException()

def check_potential_CE(x):
    u = nn.evaluate(x)
    if(np.argmin(u) != 0 ):
        print("Potential CE success")
        print(u)
        return True
    return False


def run_instance(nn, input_bounds, check_property, adv_found, target):
    nn.set_bounds(input_bounds)
    if np.min(nn.layers[7]['conc_lb'][1:]) > nn.layers[7]['conc_ub'][0]:
        print("Problem Infeasible")
        return
    solver = Solver(network = nn,property_check=check_property)
    #Add Input bounds as constraints in the solver
    #TODO: Make the solver apply the bound directly from the NN object
    input_vars = [solver.state_vars[i] for i in range(len(solver.state_vars))]
    A = np.eye(nn.image_size)
    lower_bound = input_bounds[:,0]
    upper_bound = input_bounds[:,1]
    solver.add_linear_constraints(A,input_vars,lower_bound,GRB.GREATER_EQUAL)
    solver.add_linear_constraints(A,input_vars,upper_bound,GRB.LESS_EQUAL)

    output_vars = [solver.out_vars[i] for i in range(len(solver.out_vars))]

    for idx,var in enumerate(output_vars):
        if(idx == target):
            continue
        A = [[1,-1]]
        out_vars = [solver.out_vars[target], var]
        b = [0] * 1
        solver.add_linear_constraints(A,out_vars,b,GRB.LESS_EQUAL)



    solver.preprocessing = False
    nn_in,nn_out,status = solver.solve()
    if(status == 'SolFound'):
        adv_found.value = 1


if __name__ == "__main__":

    #init Neural network
    TIMEOUT= 900
    network = "models/ACASXU_run2a_4_5_batch_2000.nnet"
    results = []
    start_time = time()
    unsafe = 0
    # networks = [networks[-1]]
    raw_lower_bounds = np.array([36000, 0.7, -3.141592, 900, 600]).reshape((-1,1))
    raw_upper_bounds = np.array([60760, 3.141592, -3.141592, 1200, 1200]).reshape((-1,1))

    print("Checking property 9 on %s"%network[5:])
    nnet = NeuralNetworkStruct()
    nnet.parse_network(network)
    lower_bounds = nnet.normalize_input(raw_lower_bounds)
    upper_bounds = nnet.normalize_input(raw_upper_bounds)
    # sample_network(nn,lower_bounds,upper_bounds)
    input_bounds = np.concatenate((lower_bounds,upper_bounds),axis = 1)
    other_ouputs = [i for i in range(nnet.output_size) if i != 0]
    for other_out in other_ouputs:

        problems = split_input_space(nnet,input_bounds,512)
        print(len(problems),"subproblems")
        adv_found = Value('i',0)
        processes = []
        try:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(TIMEOUT)
            for input_bounds in problems:
                nn = deepcopy(nnet)
                # input_bounds = problems[k]
                p = Process(target=run_instance, args=(nn, input_bounds, check_potential_CE,adv_found, other_out))
                p.start()
                processes.append(p)
                    
            while(any(p.is_alive() for p in processes) and adv_found.value == 0):
                pass
            if(adv_found.value == 1):
                print("Adv found")
                unsafe +=1
                results.append("UNSAFE")
                for p in processes:
                    p.terminate()
                break
            else:
                print("No Adv")
                results.append("Safe")

            
        except TimeOutException as e:
            results.append("Timeout") 
            for p in processes:
                p.terminate() 
            break
        
        for p in processes:
            p.terminate()
        # sys.exit()
    print('Total time for all nets:',time()-start_time)
    print(results)

    #active neurons [ 3  4  6  7  9 12 15 20 22 23 25 27 28 30 32 33 34 40 45 49]

