from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
import sys,os
import glob
from NeuralNetwork import *

eps = 1E-3



def check_property(x):
    u = nn.evaluate(x)
    if(np.argmin(u) == 1):
        print("Potential CE succeeded")
        return True
    return False

if __name__ == "__main__":

    #init Neural network
    nn = NeuralNetworkStruct()
    nn.parse_network('nnet/ACASXU_run2a_1_1_batch_2000.nnet')
    points = [np.array([0, 0, 0, 0, 0]) , np.array([0.2, -0.1, 0, -0.3, 0.4]),
            np.array([0.45, -0.23, -0.4, 0.12, 0.33]), np.array([-0.2, -0.25, -0.5, -0.3, -0.44 ]), np.array([0.61, 0.36, 0.0, 0.0, -0.24])]
    pairs = []
    # deltas = [1.0,0.75,0.5,0.25,0.1]
    deltas = [0.1, 0.075, 0.05, 0.025, 0.01]
    for pt in [points[0]]:
        y = nn.evaluate(pt)
        pairs.append({'x':pt.reshape((-1,1)), 'y':y.reshape((-1,1)) , 'out': np.argmin(y)})
    for pair in pairs:
        x = pair['x']
        y = pair['y']
        label = pair['out']
        other_ouputs = [i for i in range(nn.output_size) if i != label]
        # other_ouputs = [1]
        for delta in [deltas[2]]:
            #Solve the problem for each other output
            for out_idx in other_ouputs:
                
                input_bounds = np.concatenate((x-delta,x+delta),axis = 1)
                nn.set_bounds(input_bounds)
                solver = Solver(network = nn,property_check=check_property,target = out_idx)
                #Add Input bounds as constraints in the SAT solver
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
                A = np.zeros(nn.output_size)
                A[out_idx] = 1
                A[label] = -1
                b = [-eps]
                solver.add_linear_constraints([A],output_vars,b,GRB.LESS_EQUAL)


            
                solver.preprocessing = False
                s = time()
                nn_in,nn_out,status = solver.solve()
                e = time()
                if(status == 'SolFound'):
                    nn_in = np.array([solver.state_vars[idx].X for idx in range(nn.image_size)]).reshape((-1,1))
                    nn_out = np.array([solver.out_vars[idx].X for idx in range(nn.output_size)]).reshape((-1,1))
                    net_out = nn.evaluate(nn_in)
                    err = np.sum(np.fabs(nn.evaluate(nn_in) - nn_out))
                    print nn_in
                    print net_out
                    print('Adversarial example found with label %d ,delta %f in time %f'%(out_idx,delta,e-s))
                    print('Error',err)
                else:
                    print("Problem Infeasible,Time:%f"%(e-s))

                sys.exit()

            

#active neurons [ 3  4  6  7  9 12 15 20 22 23 25 27 28 30 32 33 34 40 45 49]

