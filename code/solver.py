from pulp import *
from random import random,seed
from time import time
sys.path.append('./z3/z3-4.4.1-x64-osx-10.11/bin/')
import pickle
from Workspace import Workspace
import math
from gurobipy import * 
import sys
import os
from NeuralNetwork import *
from copy import copy,deepcopy
from QuickXplain import QuickXplain
import re
import cdd
eps = 1E-5

class Solver():

    def __init__(self, network = None, target = 0,maxIter = 100000,property_check=None):
        self.maxNumberOfIterations = maxIter
        self.nn        = deepcopy(network)
        self.orig_net = deepcopy(self.nn)

        #TODO: self.__parse_network() #compute the dims of input and hidden nodes
        self.__input_dim    = self.nn.image_size
        self.__hidden_units = self.nn.num_hidden_neurons
        self.__output_dim   = self.nn.output_size
        self.num_layers     = self.nn.num_layers #including the input/output layers
        self.check_potential_CE = property_check

        self.model = Model()
        self.model.params.OutputFlag = 0
        self.model.params.DualReductions = 0
        #Add variables
        self.state_vars         = self.model.addVars(self.__input_dim,name = "x", lb  = -1*GRB.INFINITY)  
        self.relu_vars           = self.model.addVars(self.__input_dim,name = "y", lb = -1*GRB.INFINITY)      
        self.relu_vars.update(self.model.addVars([self.__input_dim + i for i in range(self.__hidden_units)],name = "y", lb = 0))
        self.net_vars        = self.model.addVars(self.__input_dim,name = "n",lb = -1* GRB.INFINITY)      
        self.net_vars.update(self.model.addVars([self.__input_dim + i for i in range(self.__hidden_units)],name = "n", lb = -1* GRB.INFINITY) )
        self.slack_vars         = self.model.addVars(self.__input_dim + self.__hidden_units,name = "s",lb = 0)
        self.out_vars           = self.model.addVars(self.__output_dim,name = "u", lb = -1* GRB.INFINITY)
        self.abs2d              = [[0,i] for i in range(self.__input_dim)]
        self._2dabs              = {}
        self.fixed_relus = set()

        #Layer index 
        self.model.update()
        self.layer_start_idx = [0] * len(self.nn.layers)
        idx = self.__input_dim
        for layer_idx, layer in self.nn.layers.items():
            if(layer_idx == 0):
                continue
            self._2dabs[layer_idx] = {}
            self.layer_start_idx[layer_idx] = self.layer_start_idx[layer_idx-1] + self.nn.layers[layer_idx-1]['num_nodes']
            for neuron_idx in range(layer['num_nodes']):
                self.abs2d += [[layer_idx,neuron_idx]]
                self._2dabs[layer_idx][neuron_idx] = idx
                idx+=1
        self.linear_constraints = []

    def add_linear_constraints(self, A, x, b, sense = GRB.LESS_EQUAL):
        #Senses are GRB.LESS_EQUAL, GRB.EQUAL, or GRB.GREATER_EQUAL
        for row in range(len(b)):
            linear_expression = LinExpr(A[row],x)
            constraint = {'expr' : linear_expression, 'sense': sense,'rhs': b[row]} 
            self.linear_constraints.append(constraint)

    def __add_NN_constraints(self):
        self.fixed_relus = set()
        fixed_relus = 0
        #First layer of network is assumed to be the input to the network
        layer_idx = 0
        num_neurons = self.nn.layers[layer_idx]['num_nodes']
        layer_start_idx = self.layer_start_idx[layer_idx]
        input_vars = [self.state_vars[i] for i in range(len(self.state_vars))]
        for neuron_idx in range(num_neurons):
            neuron_abs_idx = layer_start_idx + neuron_idx
            self.model.addConstr(self.relu_vars[neuron_abs_idx] == self.state_vars[neuron_abs_idx])
            self.model.addConstr(self.net_vars[neuron_abs_idx] == self.state_vars[neuron_abs_idx])
        for layer_idx in range(1,self.num_layers): #exclude input
            num_neurons = self.nn.layers[layer_idx]['num_nodes']
            layer_start_idx = self.layer_start_idx[layer_idx]
            prev_layer_start_idx = self.layer_start_idx[layer_idx - 1]
            W = self.nn.layers[layer_idx]['weights']
            b = self.nn.layers[layer_idx]['bias']
            lb = self.nn.layers[layer_idx]['conc_lb']
            ub = self.nn.layers[layer_idx]['conc_ub']
            in_lb = self.nn.layers[layer_idx]['in_lb']
            in_ub = self.nn.layers[layer_idx]['in_ub']

            prev_layer_size = self.nn.layers_sizes[layer_idx -1]
            prev_relus = [self.relu_vars[prev_layer_start_idx + input_idx] for input_idx in range(prev_layer_size)]
            for neuron_idx in range(num_neurons):
                #add - constraints
                neuron_abs_idx = layer_start_idx + neuron_idx
                net_expr = LinExpr(W[neuron_idx], prev_relus)
                if(self.nn.layers[layer_idx]['type'] != 'output'):
                    self.model.addConstr(self.net_vars[neuron_abs_idx] == (net_expr + b[neuron_idx]))
                    self.model.addConstr(self.slack_vars[neuron_abs_idx] == self.relu_vars[neuron_abs_idx] - self.net_vars[neuron_abs_idx])

                    if(ub[neuron_idx] <= 0):
                        self.model.addConstr(self.relu_vars[neuron_abs_idx] == 0)
                        fixed_relus +=1
                        self.fixed_relus.add(neuron_abs_idx)
                    elif(in_lb[neuron_idx] > 0):
                        self.model.addConstr(self.slack_vars[neuron_abs_idx] == 0)
                        fixed_relus +=1
                        self.fixed_relus.add(neuron_abs_idx) 
                    else:
                        factor = (in_ub[neuron_idx]/ (in_ub[neuron_idx]-in_lb[neuron_idx]))[0]
                        self.model.addConstr(self.relu_vars[neuron_abs_idx] <= factor * (self.net_vars[neuron_abs_idx]- in_lb[neuron_idx]),name="relaxed_%d"%neuron_abs_idx)
                        A_up = self.nn.layers[layer_idx]['Relu_sym'].upper[neuron_idx]
                        self.model.addConstr(LinExpr(A_up[:-1],input_vars)  + A_up[-1]  >= self.relu_vars[neuron_abs_idx])
            
                else:
                    self.model.addConstr(self.out_vars[neuron_idx] == (net_expr + b[neuron_idx]))
                    self.model.addConstr(self.out_vars[neuron_idx] >= lb[neuron_idx])
                    self.model.addConstr(self.out_vars[neuron_idx] <= ub[neuron_idx])
                    A_up = self.nn.layers[layer_idx]['Relu_sym'].upper[neuron_idx]
                    A_low = self.nn.layers[layer_idx]['Relu_sym'].lower[neuron_idx]
                    self.model.addConstr(LinExpr(A_up[:-1],input_vars)  + A_up[-1]  >= self.out_vars[neuron_idx])
                    self.model.addConstr(LinExpr(A_low[:-1],input_vars)  + A_low[-1]  <= self.out_vars[neuron_idx])

                
        # print('Number of fixed Relus:', len(self.fixed_relus))
    
    def solve(self):
        
        status = 'TLE'
        solutionFound = False
        iterationsCounter = -1
        counter_examples = []
        while solutionFound == False and iterationsCounter < self.maxNumberOfIterations:
            iterationsCounter               = iterationsCounter + 1

            if iterationsCounter % 100 == 0:
                pass
                # self.print_lock.acquire()
                # print('******** Solver , iteration = ', iterationsCounter ,'********')
                # self.print_lock.release()

            self.__prepare_problem()
            # self.model.write('model.lp')
            self.model.optimize()
            if(self.model.Status == 3): #Infeasible
                IIS_slack = []
                try:
                    self.model.computeIIS() 
                    fname = 'result.ilp'
                    self.model.write(fname)
                except Exception as e:
                    print(e)
                status = 'UNSAT'
                return None,None,status
            else:   
                status = 'UNKNOWN'
                SAT,infeasible_relus = self.check_SAT() 
                solutionFound = True
                if(SAT):
                    # print('Solution found')
                    x = [self.model.getVarByName('x[%d]'%i).X for i in range(len(self.state_vars))]
                    u = [self.model.getVarByName('u[%d]'%i).X for i in range(len(self.out_vars))]
                    # print('x',x)
                    # print('u',u)
                    status = 'SolFound'  
                    return x,u,status
                else:
                    status = 'UNKNOWN'
                    layers_masks = []
                    for layer_idx,layer in self.nn.layers.items():
                        if(layer_idx < 1):
                            continue
                        layers_masks += [-1*np.ones((layer['num_nodes'],1))]
                    for l,n in self.nn.active_relus:
                        layers_masks[l-1][n] = 1
                    for l,n in self.nn.inactive_relus:
                        layers_masks[l-1][n] = 0
                    non_lin_relus = [self._2dabs[l][n] for l,n in self.nn.nonlin_relus]

                    paths = [1]
                    status = self.dfs(infeasible_relus,[],layers_masks,undecided_relus=copy(sorted(non_lin_relus)),paths = paths)
                    # print(status)
                    # print('Paths:',paths)

        
        return self.model.getVars(),counter_examples,status
    def fix_relu(self, fixed_relus):
        input_vars = [self.state_vars[i] for i in range(len(self.state_vars))]
        for relu_idx, phase in fixed_relus:
            layer_idx,neuron_idx = self.abs2d[relu_idx]
            A_up = self.nn.layers[layer_idx]['in_sym'].upper[neuron_idx]
            A_low = A_up
            if(phase == 1):
                self.model.addConstr(self.slack_vars[relu_idx] == 0,name="active_"+str(relu_idx))
                self.model.addConstr(LinExpr(A_low[:-1],input_vars) + A_low[-1] == self.relu_vars[relu_idx],name ="y%d_active_LB"%relu_idx)
                self.model.addConstr(LinExpr(A_up[:-1],input_vars)  + A_up[-1]  >= 0,name ="y%d_active_LB"%relu_idx)
            else:
                self.model.addConstr(self.relu_vars[relu_idx] == 0,name="inactive_"+str(relu_idx))
                # self.model.addConstr(LinExpr(A_low[:-1],input_vars) + A_low[-1] <= 0,name ="y%d_inactive_LB"%relu_idx)
                self.model.addConstr(LinExpr(A_up[:-1],input_vars)  + A_up[-1]  <= 0,name ="y%d_inactive_UB"%relu_idx)
        
        self.add_objective([idx for idx,_ in fixed_relus])

    def update_in_interval(self):
        H_rep = np.zeros((0,6))
        for layer_idx, neuron_idx in self.nn.active_relus:
            eq = self.nn.layers[layer_idx]['in_sym'].upper[neuron_idx]
            b,A = -eq[-1], eq[:-1]
            H_rep = np.vstack((H_rep,np.hstack((-b,A))))
        try:
            for layer_idx, neuron_idx in self.nn.inactive_relus:
                eq = self.nn.layers[layer_idx]['in_sym'].upper[neuron_idx]
                b,A = -eq[-1], eq[:-1]
                H_rep = np.concatenate((H_rep,np.hstack((b,-A)).reshape((1,6))),axis = 0)

            A = cdd.Matrix(H_rep)
            A.rep_type = 1
            p = cdd.Polyhedron(A)
        
            vertices = np.array(p.get_generators())[:,1:]
            hrect_min = np.min(vertices,axis = 0).reshape((-1,1))
            hrect_max = np.max(vertices,axis = 0).reshape((-1,1))
            new_bound = np.hstack((hrect_min,hrect_max))
            new_bound[:,1] = np.minimum(new_bound[:,1],self.orig_net.input_bound[:,1])
            new_bound[:,0] = np.maximum(new_bound[:,0],self.orig_net.input_bound[:,0])
        except Exception as e:
            new_bound = self.nn.input_bound       

        return new_bound

    def set_neuron_bounds(self,layer_idx,neuron_idx,phase,layers_masks,bounds = None):
        if(phase == 0):
            layers_masks[layer_idx-1][neuron_idx] = 0
            # self.nn.update_bounds(layer_idx,neuron_idx,[np.array(0),np.array(0)],layers_masks)
        elif(phase == 1):
            layers_masks[layer_idx-1][neuron_idx] = 1
            # self.nn.update_bounds(layer_idx,neuron_idx,bounds,layers_masks)
            
        else:
            layers_masks[layer_idx-1][neuron_idx] = -1

        self.nn.recompute_bounds(layers_masks)
        # bounds = self.update_in_interval()
        # self.nn.input_bound = bounds
        # self.nn.recompute_bounds(layers_masks)
        # self.nn.input_bound = copy(self.orig_net.input_bound)

    def getIIS(self,fname):
        IIS = []
        self.model.computeIIS()
        fname = 'result1.ilp'
        self.model.write(fname)
        with open(fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if('B:' in line):
                    IIS.append(int(line.strip().split('_')[0][1:]))
        return IIS  

    def split_neuron(self,infeas_relus,max_neuron):
        in_rectangle = self.update_in_interval()
        in_rectangle = self.orig_net.input_bound
        A = np.eye(5)
        upper_bound = self.nn.input_bound[:,1].reshape((-1,1))
        lower_bound = self.nn.input_bound[:,0].reshape((-1,1))
        H_rep = np.hstack((upper_bound,-A))
        H_rep = np.vstack((H_rep,np.hstack((-lower_bound,A))))
        min_volume = np.prod(in_rectangle[:,1] - in_rectangle[:,0])
        ret,min_phase = infeas_relus[0]

        for relu_idx,_ in infeas_relus:
        # for relu_idx in infeas_relus:
            for phase in [0,1]:
                try:
                    if(relu_idx >= max_neuron):
                        break
                    layer_idx, neuron_idx = self.abs2d[relu_idx]
                    eq = self.nn.layers[layer_idx]['in_sym'].upper[neuron_idx]
                    b,A = -eq[-1], eq[:-1]
                    if(phase == 0):
                        A_matrix = np.vstack((H_rep,np.hstack((b,-A))))   
                    else:
                        A_matrix = np.vstack((H_rep,np.hstack((-b,A))))           
                    H = cdd.Matrix(A_matrix)   
                    H.rep_type = 1
                    p = cdd.Polyhedron(H)
                    vertices = np.array(p.get_generators())[:,1:]
                    hrect_min = np.min(vertices,axis = 0).reshape((-1,1))
                    hrect_max = np.max(vertices,axis = 0).reshape((-1,1))
                    new_bound = np.hstack((hrect_min,hrect_max))
                    new_bound[:,1] = np.minimum(new_bound[:,1],in_rectangle[:,1])
                    new_bound[:,0] = np.maximum(new_bound[:,0],in_rectangle[:,0])
                    volume = np.prod(new_bound[:,1] - new_bound[:,0])
                    if(volume < min_volume):
                        ret = relu_idx 
                        min_phase = phase
                        min_volume = volume
                except Exception as e:
                    pass
        return ret,min_phase

    def dfs(self, infeasible_relus,fixed_relus,layers_masks, depth = 0,undecided_relus = [],paths = 0):
        #node to be handled
        status = 'UNKNOWN'
        relu_idx,phase =  infeasible_relus[0]
        nonlin_relus = copy(undecided_relus)
        min_layer,_ = self.abs2d[nonlin_relus[0]]
        # relu_idx,phase = self.split_neuron(infeasible_relus,min_layer*50 +5)
        layer_idx,neuron_idx = self.abs2d[relu_idx]
        if(layer_idx > min_layer):
            relu_idx,phase = nonlin_relus[0],int(self.model.getVarByName('n[%d]'%nonlin_relus[0]).X >=0)
            layer_idx,neuron_idx = self.abs2d[relu_idx]
        nonlin_relus.remove(relu_idx)
        # print('DFS:',depth,"Setting neuron %d to %d"%(relu_idx,phase))
        layers_masks = deepcopy(layers_masks)
        self.set_neuron_bounds(layer_idx,neuron_idx,phase,layers_masks)
        fixed_relus.append([relu_idx,phase])
        # s = time()
        self.__prepare_problem()
        # print('Prep Problem',time() - s)
        self.fix_relu(fixed_relus)
        self.model.optimize()
        if(self.model.Status != 3): #Feasible solution
            SAT,infeasible_set = self.check_SAT()
            valid = self.check_potential_CE(np.array([self.model.getVarByName('x[%d]'%i).X for i in range(len(self.state_vars))]).reshape((-1,1)))
            if(SAT or valid):
                # print('Solution found')
                status = 'SolFound'  
            else:
                status = self.dfs(infeasible_set,copy(fixed_relus),layers_masks,depth+1,nonlin_relus,paths)
        if(status != 'SolFound'):
            paths[0] += 1 
            if(self.model.Status == 3):
                IIS = self.getIIS('result1.ilp')
                if(len(IIS) and relu_idx != IIS[-1] and IIS[-1] in [n_idx for n_idx,_ in fixed_relus]):
                    self.set_neuron_bounds(layer_idx,neuron_idx,-1,layers_masks)
                    return status

            # infeasible solution
            # set the neuron to other phase
            # qx = QuickXplain(self.quickXplain_predicate)
            # X1 = qx.quickxplain1(fixed_relus)
            # if self.model.Status == 3:
            #     X = qx.quickxplain([],fixed_relus)
            #     pass
            phase = 1 - phase
            # print('Backtrack, Setting neuron %d to %d'%(relu_idx,phase))
            self.set_neuron_bounds(layer_idx,neuron_idx,phase,layers_masks)
            fixed_relus[-1] = [relu_idx,phase]
            self.__prepare_problem()
            self.fix_relu(fixed_relus)
            self.model.optimize()
            if(self.model.Status != 3): #Feasible solution
                SAT,infeasible_set = self.check_SAT()
                valid = self.check_potential_CE(np.array([self.model.getVarByName('x[%d]'%i).X for i in range(len(self.state_vars))]).reshape((-1,1)))
                if(SAT or valid):
                    print('Solution found')
                    status = 'SolFound'  
                else:
                    status = self.dfs(infeasible_set,copy(fixed_relus),layers_masks,depth +1,nonlin_relus,paths)
            else:
                status = 'UNSAT'

            if(status != 'SolFound'):
                status = 'UNSAT'
        
            self.set_neuron_bounds(layer_idx,neuron_idx,-1,layers_masks)
        return status
            



    def check_SAT(self):   
        y = np.array([self.model.getVarByName('y[%d]'%idx).X for idx in range(self.__input_dim,len(self.relu_vars))])
        net = np.array([self.model.getVarByName('n[%d]'%idx).X for idx in range(self.__input_dim,len(self.net_vars))])
        slacks = np.zeros_like(y)
        active_infeas = ((y-net) > eps) * (net > eps) #if y>net in net>0 domain
        inactive_infeas =  ((y > eps) * (net < eps))    #if y > 0 in net<0 domain
        active  = np.sort(np.where(active_infeas == True)[0])
        inactive = np.sort(np.where(inactive_infeas == True)[0])
        slacks[active] = y[active] - net[active]
        slacks[inactive] = y[inactive]
        offset = 0
        infeas_relus=[]
        # for layer_size in self.nn.layers_sizes[1:-1]:
        #     active = active[active >= offset]
        #     inactive = inactive[inactive >= offset]
        #     layer_infeas = active[active < offset + layer_size]
        #     layer_infeas = np.concatenate((layer_infeas,inactive[inactive < offset + layer_size]))
        #     slack_infeas = slacks[layer_infeas]
        #     order_infeas = np.flip(np.argsort(slack_infeas))
        #     # order_infeas = np.argsort(slack_infeas)
        #     layer_infeas = layer_infeas[order_infeas]
        #     for neuron in layer_infeas:
        #         infeas_relus.append((neuron+self.__input_dim,int(net[neuron] > eps)))
        #     offset += layer_size
    
    
        active = list(np.where(active_infeas == True)[0] + self.__input_dim)
        inactive = list(np.where(inactive_infeas == True)[0] + self.__input_dim)            
        infeas_relus = [(n_idx,0) for n_idx in inactive]
        infeas_relus +=  [(n_idx,1) for n_idx in active]
        infeas_relus = sorted(infeas_relus)
        if(len(infeas_relus) is not 0):
            infeas_relus = [(idx,phase) for idx,phase in infeas_relus if idx not in self.fixed_relus]
            return False, infeas_relus
        return True, None

    def quickXplain_predicate(self,constraints):
        fixed_relus = sorted((constraints))
        nn = deepcopy(self.orig_net)
        layers_masks = []
        for layer_idx,layer in self.nn.layers.items():
            if(layer_idx < 1):
                continue
            layers_masks += [-1*np.ones((layer['num_nodes'],1))]
        for relu,phase in fixed_relus:
            layer_idx,relu_idx = self.abs2d[relu]
            layers_masks[layer_idx-1][relu_idx] = phase
        nn.recompute_bounds(layers_masks)
        self.nn = nn
        self.__prepare_problem()
        self.fix_relu(fixed_relus)
        self.model.optimize()
        if(self.model.Status == 2):
            return True
        return False

    def __prepare_problem(self):
        #clear all constraints
        self.model.remove(self.model.getConstrs())
        #Add external convex constraints
        for constraint in self.linear_constraints:
            self.model.addConstr(constraint['expr'], sense = constraint['sense'], rhs = constraint['rhs'])

        self.__add_NN_constraints()
        self.add_objective([])

    

    def add_objective(self, fixed_relus = None):
        slacks = self.slack_vars.values()[self.__input_dim:]
        relus  = self.relu_vars.values()[self.__input_dim:]
        slack_strt_idx = 0
        init_weight = 1E-10
        weights = []
        for layer_idx,layer_size in enumerate(self.nn.layers_sizes[1:-1]):
            ub = np.maximum(0,self.nn.layers[layer_idx+1]['in_ub'])
            ub[ub > 0] = 1
            weights += list(init_weight * ub)
            # weights += [1] * layer_size
            init_weight *= 1000

        obj = LinExpr()
        if(fixed_relus):
            for idx in fixed_relus:
                weights[idx - self.__input_dim] = 0

        obj.addTerms(weights,slacks)
        self.model.setObjective(obj)
        self.model.update()
        self.fixed_relus.update(fixed_relus)


        
    
# layers_sizes = [2,3,1]
# image_size = layers_sizes[0]
# x = np.zeros((2,1))
# bounds = np.concatenate((x,x),axis = 1)
# nn = NeuralNetworkStruct(layers_sizes,input_bounds = bounds)
# solver = Solver(network = nn)
# A = np.eye(2)
# b = np.zeros(2)
# state_vars = [solver.state_vars[0],solver.state_vars[1]]
# solver.add_linear_constraints(A,state_vars,b,LpConstraintEQ)
# A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
# b = [1,-0.1,1,-0.1]
# state_vars = [solver.state_vars[0],solver.state_vars[1]]
# solver.add_linear_constraints(A,state_vars,b)
# state_vars = [solver.out_vars[0]]
# A, b = [[-1]],[-0.1]
# solver.add_linear_constraints(A, state_vars, b)
# solver.solve()

# e = 0.1
# layers_sizes = [1,2,1]
# image_size = layers_sizes[0]
# bounds = np.zeros((1,2))
# bounds[:,1] = 1
# nn = NeuralNetworkStruct(layers_sizes,input_bounds = bounds)
# Weights= [np.concatenate((np.array([-1]),np.array([1])),axis = 0).reshape((2,1))]
# Weights.append(np.concatenate((np.array([[1],[1]])),axis = 0).reshape((1,2)))
# biases = [np.array([e,e-1]),np.zeros(2)]
# nn.set_weights(Weights,biases)
# solver = Solver(network = nn)
# state_vars = [solver.state_vars[0]]
# A, b = [[1],[-1]],[1,0]
# solver.add_linear_constraints(A, state_vars, b)
# state_vars = [solver.out_vars[0]]
# A, b = [[1],[-1]],[e,-e/2]
# solver.add_linear_constraints(A, state_vars, b)
# solver.solve()