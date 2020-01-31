from solver import *
from time import time,sleep
from random import random, seed
import numpy as np
import signal
from keras.models import load_model
from StateSpacePartitioner import StateSpacePartitioner
import pandas as pd
import sys,os
from multiprocessing import Queue,Lock,Pool,current_process,Manager
import glob
from NeuralNetwork import *
sys.path.append('./z3/z3-4.4.1-x64-osx-10.11/bin/')
import z3 as z3
eps = 1E-3


x = np.zeros((5,1))
bounds = np.concatenate((x,x),axis = 1)
nn = NeuralNetworkStruct(input_bounds=bounds)
nn.parse_network('ACASXU_run2a_1_1_batch_2000.nnet','Weights.npy')
out = nn.evaluate(np.array([0,0,0,0,0]))
solver = Solver(network = nn)
A = np.eye(5)
b = np.zeros(5)
state_vars = [solver.state_vars[0],solver.state_vars[1]]
solver.add_linear_constraints(A,state_vars,b,LpConstraintEQ)
solver.preprocessing = False
vars,_,_ = solver.solve()
