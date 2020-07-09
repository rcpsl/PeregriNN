from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *
from utils.sample_network import * 
from multiprocessing import Process, Value

eps = 1E-3

class TimeOutException(Exception):
    pass
def alarm_handler(signum, frame):
    print('TIMEOUT!')
    raise TimeOutException()

def run_instance(nn, input_bounds, check_property, adv_found):
    nn.set_bounds(input_bounds)
    if nn.layers[7]['conc_ub'][0] < COC:
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
    A = np.zeros(nn.output_size).reshape((1,-1))
    A[0,0] = 1
    b = [COC]
    # b = [0]
    solver.add_linear_constraints(A,output_vars,b,GRB.GREATER_EQUAL)
    solver.preprocessing = False
    nn_in,nn_out,status = solver.solve()
    if(status == 'SolFound'):
        adv_found.value = 1

def check_potential_CE(x):
    u = nn.evaluate(x)
    if(u[0] - COC >= eps):
        return True
    return False

if __name__ == "__main__":

    #init Neural network
    networks = sorted(glob.glob("models/ACAS*.nnet"))
    results = []
    start_time = time()
    TIMEOUT = 900
    raw_lower_bounds = np.array([55947.691, -3.141592, -3.141592, 1145, 0]).reshape((-1,1))
    raw_upper_bounds = np.array([62000, 3.141592, 3.141592, 1200, 60]).reshape((-1,1))    
    raw_COC = 1500
    for network in networks:
        print("Checking property 1 on %s"%network[5:])
        nn = NeuralNetworkStruct()
        nn.parse_network(network)
        lower_bounds = nn.normalize_input(raw_lower_bounds)
        upper_bounds = nn.normalize_input(raw_upper_bounds)
        COC = nn.normalize_output(raw_COC)
        input_bounds = np.concatenate((lower_bounds,upper_bounds),axis = 1)
        nn.set_bounds(input_bounds)
        if nn.layers[7]['conc_ub'][0] < COC:
            print("Problem Infeasible")
            continue
        # nn.layers[7]['bias'][0] -= 3.9911256459
        problems = split_input_space(network,input_bounds,128)
        print(len(problems),"subproblems")
        adv_found = Value('i',0)
        processes = []
        try:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(TIMEOUT)
            for input_bounds in problems:
                nn = deepcopy(nn)
                p = Process(target=run_instance, args=(nn, input_bounds, check_potential_CE,adv_found))
                p.start()
                processes.append(p)
                    
            while(any(p.is_alive() for p in processes) and adv_found.value == 0):
                pass
            if(adv_found.value == 1):
                print("Adv found")
                unsafe +=1
                results.append("UNSAFE")
            else:
                print("No Adv")
                results.append("Safe")

           
        except TimeOutException as e:
            results.append("Timeout")  
            
        for p in processes:
            p.terminate()

        
    print('Total time for all nets:',time()-start_time)
    print(results)
