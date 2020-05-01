from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *

eps = 1E-3



def check_potential_CE(x):
    u = nn.evaluate(x)
    if(u[0] - 3.9911256459 >= eps):
        return True
    return False

if __name__ == "__main__":

    #init Neural network
    networks = sorted(glob.glob("nnet/*.nnet"))
    results = []
    start_time = time()
    for network in networks:
        print("Checking property 1 on %s"%network[5:])
        nn = NeuralNetworkStruct()
        nn.parse_network(network)
        
        lower_bounds = np.array([0.6,-0.5,-0.5,0.45,-0.5]).reshape((-1,1))
        upper_bounds = np.array([0.6798577687,0.5,0.5,0.5,-0.45]).reshape((-1,1))
        input_bounds = np.concatenate((lower_bounds,upper_bounds),axis = 1)
        # nn.layers[7]['bias'][0] -= 3.9911256459
        nn.set_bounds(input_bounds)

        solver = Solver(network = nn,property_check=check_potential_CE)
        #Add Input bounds as constraints in the solver
        #TODO: Make the solver apply the bound directly from the NN object
        input_vars = [solver.state_vars[i] for i in range(len(solver.state_vars))]
        A = np.eye(nn.image_size)
        lower_bound = input_bounds[:,0]
        upper_bound = input_bounds[:,1]
        solver.add_linear_constraints(A,input_vars,lower_bound,GRB.GREATER_EQUAL)
        solver.add_linear_constraints(A,input_vars,upper_bound,GRB.LESS_EQUAL)

        # A = np.eye(len(solver.state_vars))
        # b = [-0.277091,0.173774,0.515735,0.978737,0.684880]
        # b = [-0.258785, 0.143822, 0.148294,0.50000,0.477025]
        # b = [-0.100000,-0.025285,0.011807,-0.009691,-0.100000]
        # solver.add_linear_constraints(A,input_vars,b,GRB.EQUAL)

        # output_vars = [solver.out_vars[i] for i in range(len(solver.out_vars))]
        # A = np.eye(len(solver.out_vars))
        # b = [-0.0162349481, -0.0180076580, -0.0178982665, -0.0178564177, -0.0174600866]
        # solver.add_linear_constraints(A,output_vars,b,GRB.EQUAL)

        output_vars = [solver.out_vars[i] for i in range(len(solver.out_vars))]
        A = np.zeros(nn.output_size).reshape((1,-1))
        A[0,0] = 1
        b = [3.9911256459]
        # b = [0]
        solver.add_linear_constraints(A,output_vars,b,GRB.GREATER_EQUAL)



        solver.preprocessing = False
        s = time()
        nn_in,nn_out,status = solver.solve()
        e = time()
        results.append(status)
        if(status == 'SolFound'):
            nn_in = np.array([solver.state_vars[idx].X for idx in range(nn.image_size)]).reshape((-1,1))
            nn_out = np.array([solver.out_vars[idx].X for idx in range(nn.output_size)]).reshape((-1,1))
            err = np.sum(np.fabs(nn.evaluate(nn_in) - nn_out))
            print(nn_in)
            print(nn_out)
            print('Adversarial example found with label %d ,delta %f in time %f'%(out_idx,delta,e-s))
            print('Error',err)
        else:
            print("Problem Infeasible,Time:%f"%(e-s))
        print("=========================")
        sys.exit()
    print(results,time()-start_time)



    #active neurons [ 3  4  6  7  9 12 15 20 22 23 25 27 28 30 32 33 34 40 45 49]

