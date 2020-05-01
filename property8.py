from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *

eps = 0



def check_potential_CE(x):
    u = nn.evaluate(x)
    if(np.argmin(u) not in [0,1]):
        print('Potential CE succeeded')
        return True
    return False

if __name__ == "__main__":

    #init Neural network
    network = "nnet/ACASXU_run2a_2_9_batch_2000.nnet"
    results = []
    print("Checking property 8 on %s"%network[5:])
    nn = NeuralNetworkStruct()
    nn.parse_network(network)
    lower_bounds = np.array([-0.3284228772,-0.5,-0.0159154943,-0.0454545455,0.0]).reshape((-1,1))
    upper_bounds = np.array([0.6798577687,-0.3749999220,0.0159154943,0.5,0.5]).reshape((-1,1))
                    
    
    input_bounds = np.concatenate((lower_bounds,upper_bounds),axis = 1)
    nn.set_bounds(input_bounds)
    start_time = time()
    for out_idx in [3]:
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
        solver.add_linear_constraints([[-1,0,0,1,0]],output_vars,[-eps],GRB.LESS_EQUAL)
        solver.add_linear_constraints([[0,-1,0,1,0]],output_vars,[-eps],GRB.LESS_EQUAL)

        # A = np.eye(len(solver.state_vars))
        # b = [0.159963, -0.498047, -0.009947, 0.201705, 0.246094 ]
        # solver.add_linear_constraints(A,input_vars,b,GRB.EQUAL)
        solver.preprocessing = False
        nn_in,nn_out,status = solver.solve()
        e = time()
        results.append(status)
        if(status == 'SolFound'):
            nn_in = np.array([solver.state_vars[idx].X for idx in range(nn.image_size)]).reshape((-1,1))
            nn_out = np.array([solver.out_vars[idx].X for idx in range(nn.output_size)]).reshape((-1,1))
            err = np.sum(np.fabs(nn.evaluate(nn_in) - nn_out))
            print(nn_in)
            print (nn_out)
            print('Counter example found with label %d, in time %f'%(out_idx,e-start_time))
            print('Error',err)
            break
        else:
            print("Problem Infeasible,Time:%f"%(e-start_time))
            del solver
        print("=========================")
    print(results,time()-start_time)




