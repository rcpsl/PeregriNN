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
    # print('TIMEOUT!')
    raise TimeOutException()

def run_instance(nn, input_bounds, check_property, adv_found):
    nn.set_bounds(input_bounds)
    if nn.layers[7]['conc_ub'][0] < COC:
        # print("Problem Infeasible")
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
    if(len(sys.argv) > 1):
        networks = ['models/ACASXU_run2a_4_6_batch_2000.nnet','models/ACASXU_run2a_4_8_batch_2000.nnet']
        TIMEOUT = 21600
        split_input = split_input_space1
    else:
        networks = sorted(glob.glob("models/ACAS*.nnet"))
        TIMEOUT = 300
        split_input = split_input_space
    results = []
    start_time = time()
    
    raw_lower_bounds = np.array([55947.691, -3.141592, -3.141592, 1145, 0]).reshape((-1,1))
    raw_upper_bounds = np.array([62000, 3.141592, 3.141592, 1200, 60]).reshape((-1,1))    
    raw_COC = 1500
    for network in networks:
        instance_start = time()
        # print("Checking property 1 on %s"%network[5:])
        nn = NeuralNetworkStruct()
        nn.parse_network(network)
        lower_bounds = nn.normalize_input(raw_lower_bounds)
        upper_bounds = nn.normalize_input(raw_upper_bounds)
        COC = nn.normalize_output(raw_COC)
        input_bounds = np.concatenate((lower_bounds,upper_bounds),axis = 1)
        nn.set_bounds(input_bounds)
        if nn.layers[7]['conc_ub'][0] < COC:
            # print("Problem Infeasible")
            print_summary(network,1,'safe',time()-instance_start)
            continue
        # nn.layers[7]['bias'][0] -= 3.9911256459
        problems = split_input(nn,input_bounds,512)
        # print(len(problems),"subproblems")
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

            prev_n_alive = -1       
            while(any(p.is_alive() for p in processes) and adv_found.value == 0):
                sleep(5)
                n_alive = np.sum([p.is_alive() for p in processes])
                if(n_alive != prev_n_alive):
                    prev_n_alive = n_alive
                    # print('Progress %d/%d' %(len(problems)-n_alive,len(problems)))
             
            if(adv_found.value == 1):
                # print("Adv found")
                unsafe +=1
                results.append("UNSAFE")
                print_summary(network,1,'safe',time()-instance_start)
            else:
                # print("No Adv")
                results.append("Safe")
                print_summary(network,1,'safe',time()-instance_start)

           
        except TimeOutException as e:
            print_summary(network,1,'timeout',TIMEOUT)
            results.append("Timeout") 
            for p in processes:
                p.terminate()  

        for p in processes:
            p.terminate()

        
    # print('Total time:',time()-start_time)
    # print(results)
