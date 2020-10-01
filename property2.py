from solver import *
from time import time,sleep
from random import random, seed, uniform
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *
from utils.sample_network import * 
from utils.utils import *
from multiprocessing import Process,Value
from hitandrun import *
eps = 1E-3

class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    # print('TIMEOUT!')
    raise TimeOutException()

def check_potential_CE(x):
    u = nn.evaluate(x)
    if(np.argmax(u) == 0 ):
        # print("Potential CE success")
        return True
    return False

def check_prop_samples(nn,samples):
    _,outs = nn.get_phases(samples)
    outs = np.argmax(outs,axis = 1)
    return np.any(outs  == 0)

def run_instance(nn, input_bounds, check_property, adv_found):
    nn.set_bounds(input_bounds)
    if np.max(nn.layers[7]['conc_lb'][1:]) > nn.layers[7]['conc_ub'][0]:
        # print("Problem Infeasible")
        return
    solver = Solver(network = nn,property_check=check_property, samples = samples)
    #Add Input bounds as constraints in the solver
    #TODO: Make the solver apply the bound directly from the NN object
    A = np.eye(nn.image_size)
    lower_bound = input_bounds[:,0]
    upper_bound = input_bounds[:,1]
    solver.add_linear_constraints(A,solver.in_vars_names,lower_bound,GRB.GREATER_EQUAL)
    solver.add_linear_constraints(A,solver.in_vars_names,upper_bound,GRB.LESS_EQUAL)

    
    A = [[1,-1,0,0,0],[1,0,-1,0,0],[1,0,0,-1,0],[1,0,0,0,-1]]
    b = [0] * 4
    solver.add_linear_constraints(A,solver.out_vars_names,b,GRB.GREATER_EQUAL)



    solver.preprocessing = False
    nn_in,nn_out,status = solver.solve()
    if(status == 'SolFound'):
        adv_found.value = 1


if __name__ == "__main__":

    #init Neural network
    if(len(sys.argv) > 1):
        networks = ['models/ACASXU_run2a_3_3_batch_2000.nnet','models/ACASXU_run2a_4_2_batch_2000.nnet',
                    'models/ACASXU_run2a_4_9_batch_2000.nnet','models/ACASXU_run2a_5_3_batch_2000.nnet']
        TIMEOUT = 21600
        split_input = split_input_space1
    else:
        networks = sorted(glob.glob("models/ACAS*2a_[1-5]_*.nnet"))
        TIMEOUT = 300
        split_input = split_input_space

    results = []
    start_time = time()
    unsafe = 0
    # networks = [networks[-1]]
    raw_lower_bounds = np.array([55947.691, -3.141592, -3.141592, 1145, 0]).reshape((-1,1))
    raw_upper_bounds = np.array([62000, 3.141592, 3.141592, 1200, 60]).reshape((-1,1))
    for network in networks:
        instance_start = time()
        # print("Checking property 2 on %s"%network[5:])
        nnet = NeuralNetworkStruct()
        nnet.parse_network(network)
        lower_bounds = nnet.normalize_input(raw_lower_bounds)
        upper_bounds = nnet.normalize_input(raw_upper_bounds)
        input_bounds = np.concatenate((lower_bounds,upper_bounds),axis = 1)
        nnet.set_bounds(input_bounds)
        if np.max(nnet.layers[7]['conc_lb'][1:]) > nnet.layers[7]['conc_ub'][0]:
            # print("Problem Infeasible")
            print_summary(network,2,'safe',time()-instance_start)
            continue
        samples = sample_network(nnet,input_bounds,15000)
        SAT = check_prop_samples(nnet,samples)
        if(SAT):
            print_summary(network,2,'unsafe using samples',time()-instance_start)
            continue
        problems = split_input(nnet,input_bounds,512)
        # problems = [input_bounds]
        # print(len(problems),"subproblems")
        adv_found = Value('i',0)
        processes = []
        try:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(TIMEOUT)
            for input_bounds in problems:
                nn = deepcopy(nnet)
                samples = sample_network(nnet,input_bounds,15000)
                # input_bounds = problems[k]
                # run_instance(nn, input_bounds, check_potential_CE,adv_found)
                p = Process(target=run_instance, args=(nn, input_bounds, check_potential_CE,adv_found))
                p.start()
                processes.append(p)
            prev_n_alive = -1        
            while(any(p.is_alive() for p in processes) and adv_found.value == 0):
                sleep(5)
                n_alive = np.sum([p.is_alive() for p in processes])
                if(n_alive != prev_n_alive):
                    prev_n_alive = n_alive
                    #print('Progress %d/%d' %(len(problems)-n_alive,len(problems)))
                    
            if(adv_found.value == 1):
                # print("Adv found")
                unsafe +=1
                results.append("UNSAFE")
                print_summary(network,2,'unsafe',time()-instance_start)
            else:
                # print("No Adv")
                results.append("Safe")
                print_summary(network,2,'safe',time()-instance_start)

           
        except TimeOutException as e:
            print_summary(network,2,'timeout',time()-instance_start)
            results.append("Timeout") 
            for p in processes:
                p.terminate()   
            
        for p in processes:
            p.terminate()
        # sys.exit()
    # print('Total time for all nets:',time()-start_time)
    # print(results)

    #active neurons [ 3  4  6  7  9 12 15 20 22 23 25 27 28 30 32 33 34 40 45 49]

