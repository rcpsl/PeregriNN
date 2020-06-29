from solver import *
from time import time,sleep
from random import random, seed, uniform
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *
from utils.sample_network import * 

eps = 1E-3



def check_potential_CE(x):
    u,_ = nn.evaluate(x)
    if(np.argmax(u) == 0 ):
        print("Potential CE success")
        return True
    return False




if __name__ == "__main__":

    #init Neural network
    networks = sorted(glob.glob("models/ACAS*2a_2_*.nnet"))
    results = []
    start_time = time()
    unsafe = 0
    for network in networks:
        print("Checking property 1 on %s"%network[5:])
        nnet = NeuralNetworkStruct()
        nnet.parse_network(network)
        lower_bounds = np.array([0.6,-0.5,-0.5,0.45,-0.5]).reshape((-1,1))
        upper_bounds = np.array([0.6798577687,0.5,0.5,0.5,-0.45]).reshape((-1,1))
        # lower_bounds = np.array([0.600000, -0.500000, -0.500000, 0.450000, -0.500000]).reshape((-1,1))
        # upper_bounds = np.array([0.679858, -0.250000, -0.375000, 0.500000, -0.450000]).reshape((-1,1))

        # sample_network(nn,lower_bounds,upper_bounds)
        input_bounds = np.concatenate((lower_bounds,upper_bounds),axis = 1)
        problems = split_input_space(nnet,input_bounds)
        for _,input_bounds in problems:
            nn = deepcopy(nnet)
            # input_bounds = problems[k]
            nn.set_bounds(input_bounds)
            if np.max(nn.layers[7]['conc_lb'][1:]) > nn.layers[7]['conc_ub'][0]:
                print("Problem Infeasible")
                continue
            solver = Solver(network = nn,property_check=check_potential_CE)
            #Add Input bounds as constraints in the solver
            #TODO: Make the solver apply the bound directly from the NN object
            input_vars = [solver.state_vars[i] for i in range(len(solver.state_vars))]
            A = np.eye(nn.image_size)
            lower_bound = input_bounds[:,0]
            upper_bound = input_bounds[:,1]
            solver.add_linear_constraints(A,input_vars,lower_bound,GRB.GREATER_EQUAL)
            solver.add_linear_constraints(A,input_vars,upper_bound,GRB.LESS_EQUAL)

            output_vars = [solver.out_vars[i] for i in range(len(solver.out_vars))]
            A = [[1,-1,0,0,0],[1,0,-1,0,0],[1,0,0,-1,0],[1,0,0,0,-1]]
            b = [0] * 4
            solver.add_linear_constraints(A,output_vars,b,GRB.GREATER_EQUAL)



            solver.preprocessing = False
            s = time()
            nn_in,nn_out,status = solver.solve()
            e = time()
            results.append(status)
            if(status == 'SolFound'):
                unsafe +=1
                nn_in = np.array([solver.state_vars[idx].X for idx in range(nn.image_size)]).reshape((-1,1))
                nn_out = np.array([solver.out_vars[idx].X for idx in range(nn.output_size)]).reshape((-1,1))
                u,_= nn.evaluate(nn_in)
                err = np.sum(np.fabs( u- nn_out))
                print(nn_in)
                print(nn_out)
                # print('Adversarial example found with label %d ,delta %f in time %f'%(out_idx,delta,e-s))
                print('Error',err)
                break
            else:
                print("Problem Infeasible,Time:%f"%(e-s))
        print("=========================")
        print(results,time()-start_time)
        # sys.exit()
    print('Total time for all nets:',time()-start_time)
    print('UNSAFE', unsafe)


    #active neurons [ 3  4  6  7  9 12 15 20 22 23 25 27 28 30 32 33 34 40 45 49]

