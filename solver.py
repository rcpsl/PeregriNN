from random import random,seed
from time import time
import pickle
import math
from gurobipy import * 
import sys
import os
from NeuralNetwork import *
from copy import copy,deepcopy
import re
import cdd
from poset import *
from utils.sample_network import *
from hitandrun import *
from polytope import *
eps = 1E-5



class Solver():

    def __init__(self, network = None, target = -1,maxIter = 100000,property_check=None, samples = None):
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
        #Variable names

        self.in_vars_names       = ['x[%d]'%i for i in range(self.__input_dim)]
        self.relu_vars_names     = ['y[%d]'%i for i in range(self.__input_dim + self.__hidden_units)]
        self.net_vars_names      = ['n[%d]'%i for i in range(self.__input_dim + self.__hidden_units)]
        self.slack_vars_names    = ['s[%d]'%i for i in range(self.__input_dim + self.__hidden_units)]
        self.out_vars_names      = ['u[%d]'%i for i in range(self.__output_dim)]

        self.abs2d              = [[0,i] for i in range(self.__input_dim)]
        self._2dabs              = {}
        self.fixed_relus = set()
        self.MAX_DEPTH = 2500
        self.samples = samples
        self.phases,self.samples_outs = self.nn.get_phases(self.samples)

        #Layer index 
        self.model.update()
        self.layer_start_idx = [0] * len(self.nn.layers)
        self.layer_stats = np.zeros((self.nn.num_layers-1,2))

        idx = self.__input_dim
        for layer_idx, layer in self.nn.layers.items():
            if(layer_idx == 0):
                continue
            # self.layer_stats[layer_idx] = {'undecided':0, 'infeasible':0}
            self._2dabs[layer_idx] = {}
            self.layer_start_idx[layer_idx] = self.layer_start_idx[layer_idx-1] + self.nn.layers[layer_idx-1]['num_nodes']
            for neuron_idx in range(layer['num_nodes']):
                self.abs2d += [[layer_idx,neuron_idx]]
                self._2dabs[layer_idx][neuron_idx] = idx
                idx+=1
        self.linear_constraints = []

        # if(target != -1):
        #     outs = self.out_vars.values()
        #     decision_var = self.model.addVar(name = 'd')
        #     self.model.addConstr(decision_var == max_(outs[:target] + outs[target+1:]))
        #     self.model.addConstr(decision_var >= 0)

    def add_linear_constraints(self, A, x, b, sense = GRB.LESS_EQUAL):
        #Senses are GRB.LESS_EQUAL, GRB.EQUAL, or GRB.GREATER_EQUAL
        for row in range(len(b)):
            # linear_expression = LinExpr(A[row],x)
            constraint = {'A' : A[row], 'x' : x, 'sense': sense,'rhs': b[row]} 
            self.linear_constraints.append(constraint)

    def __add_NN_constraints(self,model, nn):
        fixed_relus = 0
        #First layer of network is assumed to be the input to the network
        layer_idx = 0
        num_neurons = nn.layers[layer_idx]['num_nodes']
        layer_start_idx = self.layer_start_idx[layer_idx]
        state_vars = [model.getVarByName(var_name) for var_name in self.in_vars_names]
        out_vars = [model.getVarByName(var_name) for var_name in self.out_vars_names]
        relu_vars  = [model.getVarByName(var_name) for var_name in self.relu_vars_names]
        net_vars   = [model.getVarByName(var_name) for var_name in self.net_vars_names]
        slack_vars = [model.getVarByName(var_name) for var_name in self.slack_vars_names]
        for neuron_idx in range(num_neurons):
            neuron_abs_idx = layer_start_idx + neuron_idx
            model.addConstr(relu_vars[neuron_abs_idx] == state_vars[neuron_abs_idx])
            model.addConstr(net_vars[neuron_abs_idx]  == state_vars[neuron_abs_idx])
        for layer_idx in range(1,nn.num_layers): #exclude input
            num_neurons = nn.layers[layer_idx]['num_nodes']
            layer_start_idx = self.layer_start_idx[layer_idx]
            prev_layer_start_idx = self.layer_start_idx[layer_idx - 1]
            W = nn.layers[layer_idx]['weights']
            b = nn.layers[layer_idx]['bias']
            lb = nn.layers[layer_idx]['conc_lb']
            ub = nn.layers[layer_idx]['conc_ub']
            in_lb = nn.layers[layer_idx]['in_lb']
            in_ub = nn.layers[layer_idx]['in_ub']

            prev_layer_size = nn.layers_sizes[layer_idx -1]
            prev_relus = [relu_vars[prev_layer_start_idx + input_idx] for input_idx in range(prev_layer_size)]
            for neuron_idx in range(num_neurons):
                #add - constraints
                neuron_abs_idx = layer_start_idx + neuron_idx
                net_expr = LinExpr(W[neuron_idx], prev_relus)
                if(nn.layers[layer_idx]['type'] != 'output'):
                    model.addConstr(net_vars[neuron_abs_idx] == (net_expr + b[neuron_idx]))
                    model.addConstr(slack_vars[neuron_abs_idx] == relu_vars[neuron_abs_idx] - net_vars[neuron_abs_idx])

                    if(ub[neuron_idx] <= 0):
                        model.addConstr(relu_vars[neuron_abs_idx] == 0, name= "%d_active"%neuron_abs_idx)
                        fixed_relus +=1
                    elif(in_lb[neuron_idx] > 0):
                        model.addConstr(slack_vars[neuron_abs_idx] == 0, name= "%d_inactive"%neuron_abs_idx)
                        fixed_relus +=1
                    else:
                        factor = (in_ub[neuron_idx]/ (in_ub[neuron_idx]-in_lb[neuron_idx]))[0]
                        model.addConstr(relu_vars[neuron_abs_idx] <= factor * (net_vars[neuron_abs_idx]- in_lb[neuron_idx]),name="relaxed_%d"%neuron_abs_idx)
                        A_up = nn.layers[layer_idx]['Relu_sym'].upper[neuron_idx]
                        model.addConstr(LinExpr(A_up[:-1],state_vars)  + A_up[-1]  >= relu_vars[neuron_abs_idx])
            
                else:
                    model.addConstr(out_vars[neuron_idx] == (net_expr + b[neuron_idx]))
                    model.addConstr(out_vars[neuron_idx] >= lb[neuron_idx])
                    model.addConstr(out_vars[neuron_idx] <= ub[neuron_idx])
                    A_up = nn.layers[layer_idx]['Relu_sym'].upper[neuron_idx]
                    A_low = nn.layers[layer_idx]['Relu_sym'].lower[neuron_idx]
                    model.addConstr(LinExpr(A_up[:-1],state_vars)  + A_up[-1]  >= out_vars[neuron_idx])
                    model.addConstr(LinExpr(A_low[:-1],state_vars)  + A_low[-1]  <= out_vars[neuron_idx])

                
        # print('Number of fixed Relus:', len(self.fixed_relus))
    def __create_init_model(self):

        model = Model()
        model.params.OutputFlag = 0
        model.params.DualReductions = 0
        model.addVars(self.__input_dim,name = self.in_vars_names, lb  = -1*GRB.INFINITY)  
        model.addVars(self.__input_dim,name = self.relu_vars_names[:self.__input_dim], lb = -1*GRB.INFINITY)      
        model.addVars(self.__hidden_units,name = self.relu_vars_names[self.__input_dim:], lb = 0)
        model.addVars(self.__input_dim + self.__hidden_units,name = self.net_vars_names ,lb = -1* GRB.INFINITY)      
        model.addVars(self.__input_dim + self.__hidden_units,name = self.slack_vars_names,lb = 0)
        model.addVars(self.__output_dim,name = self.out_vars_names, lb = -1* GRB.INFINITY)
        model.update()

        return model

    def solve(self):
        
        #Create initial model
        model = self.__create_init_model()
        self.__prepare_problem(model,self.nn)
        # self.model.write('model.lp')
        model.optimize()
        if(model.Status == 3): #Infeasible
            # IIS_slack = []
            # try:
            #     self.model.computeIIS() 
            #     fname = 'result.ilp'
            #     self.model.write(fname)
            # except Exception as e:
            #     print(e)
            status = 'UNSAT'
            return None,None,status
        else:   
            status = 'UNKNOWN'
            SAT,infeasible_relus = self.check_SAT(model) 
            if(SAT):
                # print('Solution found')
                x = [model.getVarByName(var_name).X for var_name in self.in_vars_names]
                u = [model.getVarByName(var_name).X for var_name in self.out_vars_names]
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
                status = self.dfs(model, deepcopy(self.nn), infeasible_relus,[],layers_masks,undecided_relus=copy(sorted(non_lin_relus)),paths = paths)
                # print(self.layer_stats[0])
                # print(status)
                # print('Paths:',paths)

        
        return self.model.getVars(),status

    def fix_relu(self, model, nn, fixed_relus):
        input_vars = [model.getVarByName(var_name) for var_name in self.in_vars_names]
        for relu_idx, phase in fixed_relus:
            layer_idx,neuron_idx = self.abs2d[relu_idx]
            A_up = nn.layers[layer_idx]['in_sym'].upper[neuron_idx]
            A_low = A_up
            slack_var = model.getVarByName(self.slack_vars_names[relu_idx])
            relu_var  = model.getVarByName(self.relu_vars_names[relu_idx])
            if(phase == 1):
                model.addConstr(slack_var == 0,name="%d_active"%relu_idx)
                model.addConstr(LinExpr(A_low[:-1],input_vars) + A_low[-1] == relu_var,name ="y%d_active_LB"%relu_idx)
                model.addConstr(LinExpr(A_up[:-1],input_vars)  + A_up[-1]  >= 0,name ="y%d_active_LB"%relu_idx)
            else:
                model.addConstr(relu_var == 0,name="%d_inactive"%relu_idx)
                # self.model.addConstr(LinExpr(A_low[:-1],input_vars) + A_low[-1] <= 0,name ="y%d_inactive_LB"%relu_idx)
                model.addConstr(LinExpr(A_up[:-1],input_vars)  + A_up[-1]  <= 0,name ="y%d_inactive_UB"%relu_idx)
        
        # self.add_objective([idx for idx,_ in fixed_relus])

    def update_in_interval(self):
        H_rep = np.zeros((0,self.nn.image_size +1 ))
        for layer_idx, neuron_idx in self.nn.active_relus:
            eq = self.nn.layers[layer_idx]['in_sym'].upper[neuron_idx]
            b,A = -eq[-1], eq[:-1]
            H_rep = np.vstack((H_rep,np.hstack((-b,A))))
        try:
            for layer_idx, neuron_idx in self.nn.inactive_relus:
                eq = self.nn.layers[layer_idx]['in_sym'].upper[neuron_idx]
                b,A = -eq[-1], eq[:-1]
                H_rep = np.concatenate((H_rep,np.hstack((b,-A)).reshape((1,6))),axis = 0)
                self.MAX_DEPTH = 2

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

    def set_neuron_bounds(self,model,nn, layer_idx,neuron_idx,phase,layers_masks,bounds = None):
        if(phase == 0):
            layers_masks[layer_idx-1][neuron_idx] = 0
            # self.nn.update_bounds(layer_idx,neuron_idx,[np.array(0),np.array(0)],layers_masks)
        elif(phase == 1):
            layers_masks[layer_idx-1][neuron_idx] = 1
            # self.nn.update_bounds(layer_idx,neuron_idx,bounds,layers_masks)
            
        else:
            layers_masks[layer_idx-1][neuron_idx] = -1

        nn.recompute_bounds(layers_masks)
        self.fix_after_propgt(model,nn)
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
        A = np.eye(self.nn.image_size)
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
    def get_effective_weights(self,model, nn, W,b,layer):
        W_ = nn.layers[1]['weights']
        b_ = nn.layers[1]['bias']
        net = np.array([model.getVarByName(var_name).X for var_name in self.net_vars_names])
        phases = (net > 0).astype(int)
        layer_phases = phases[nn.image_size:nn.image_size + nn.layers_sizes[1]]
        for layer_idx in range(2,layer):
            W_l  = nn.layers[layer_idx]['weights']
            b_l  = nn.layers[layer_idx]['bias']
            W_ = np.matmul(W_l,W_ *layer_phases.reshape((-1,1)))
            b_ = W_l.dot(b_ * layer_phases.reshape((-1,1))) + b_l
            layer_start = self.layer_start_idx[layer_idx]
            layer_phases = phases[layer_start:layer_start + nn.layers_sizes[layer_idx]]
        
        W_ = np.matmul(W,W_ *layer_phases.reshape((-1,1)))
        b_ = W.dot(b_ *layer_phases.reshape((-1,1))) + b
        return W_,b_

    def pick_largest_ub(self, model, nn,samples,phases,y,neurons_idxs, fixed_neurons, output_idx):
        ret = [neurons_idxs[0][0],neurons_idxs[0][1],0]
        min_layer = neurons_idxs[0][0]
        num_samples = 100
        samples_idxs = list(range(num_samples))
        RESAMPLE = True
        if(RESAMPLE):
            A = np.zeros((0,nn.image_size))
            b = np.zeros((0,1))
            for l_idx,n_idx,phase in fixed_neurons:
                if(l_idx > min_layer):
                    break
                W,c = nn.layers[l_idx]['weights'][n_idx].reshape((1,-1)),nn.layers[l_idx]['bias'][n_idx]
                if(l_idx > 1):
                    W,c = self.get_effective_weights(model, nn, W,c,l_idx)
                c = -c
                if phase == 1:
                    W = -W
                    c = -c
                A = np.vstack((A,W))
                b = np.vstack((b,c))
            A_ = np.vstack((np.eye(A.shape[1]),-np.eye(A.shape[1])))
            b_ = np.vstack((nn.input_bound[:,1].reshape((-1,1)),-nn.input_bound[:,0].reshape((-1,1))))
            A = np.vstack((A,A_))
            b = np.vstack((b,b_))
            polytope = Polytope(A=A, b=b.flatten())
            x0 = np.array([model.getVarByName(var_name).X for var_name in self.in_vars_names])
            hitandrun = HitAndRun(polytope=polytope, starting_point=x0)
            samples = hitandrun.get_samples(n_samples=num_samples) 
            phases,y = nn.get_phases(samples)

        # phases,y = nn.get_phases(samples)
        else:
            for l_idx,n_idx,phase in fixed_neurons:
                samples_idxs = np.where(phases[l_idx-1][samples_idxs,n_idx]  == phase)[0]

        max_bound = -1E5
        for l_idx, n_idx in neurons_idxs:
            for phase in [True, False]:
                outs = y[np.where(phases[l_idx-1][samples_idxs,n_idx] == phase)[0]]
                if(outs.shape[0] > 0):
                    #count_prop = len(np.where(np.argmax(outs,axis =1)== output_idx)[0])
                    count_prop = np.max(outs[output_idx,:])
                    if(count_prop > max_bound):
                        max_bound = count_prop
                        ret = [l_idx,n_idx,phase]
        return ret

    def pick_one(self, model, nn, infeasible_relus, fixed_relus):
        #Assume infeasible_relus are sorted
        # try:
        k = 10
        slacks = np.array([model.getVarByName(var_name).X for var_name in self.slack_vars_names])
        y = np.array([model.getVarByName(var_name).X for var_name in self.slack_vars_names])
        x = np.array([model.getVarByName(var_name).X for var_name in self.in_vars_names]).reshape((-1,1))
        # inactive_infeas =  np.where(net < 0)[0] 
        slacks[y==0] = 0
        layer_infeasible = []
        min_layer,_ = self.abs2d[infeasible_relus[0]]
        # min_layer += 1
        for relu_idx in infeasible_relus:
            if(self.abs2d[relu_idx][0]  > min_layer):
                break
            layer_infeasible.append(relu_idx)
        ######## just choose based on sampling
        SAMPLING = True
        if(SAMPLING):
            
            prev_fixed_relus = [[layer_idx,relu_idx, 1] for layer_idx,relu_idx in nn.active_relus] 
            prev_fixed_relus += [[layer_idx,relu_idx, 0] for layer_idx,relu_idx in nn.inactive_relus] 
            prev_fixed_relus = sorted(prev_fixed_relus)
            pairs_idx = [self.abs2d[n_idx] for n_idx in layer_infeasible]
            l_idx,n_idx,phase = self.pick_largest_ub(model, nn, self.samples,self.phases,self.samples_outs,pairs_idx, prev_fixed_relus, 0)
            return self._2dabs[l_idx][n_idx],phase
            counts = count_changes_layer(nn,min_layer,self.phases,prev_fixed_relus)
            to_pick_from = counts[np.array(layer_infeasible) - self.layer_start_idx[min_layer]]
            relu_idx = np.argmin([(act + inact)/2 for act,inact in to_pick_from])
            phase = np.argmin(to_pick_from[relu_idx])
            return layer_infeasible[relu_idx] ,phase
        #######################################
        if(len(layer_infeasible) == 1):
            return infeasible_relus[0], 1 - infeasible_relus[0]
        layer_slacks = slacks[layer_infeasible]
        top_k = np.argsort(layer_slacks)
        top_k = top_k[-k:]
        W = nn.layers[min_layer]['weights'][top_k]
        b = nn.layers[min_layer]['bias'][top_k]
        if(min_layer > 1):
            W,b = self.get_effective_weights(W,b,min_layer)
        b = -b
        c = W.dot(x) 
        idxs = np.where(c > b)[0]
        W[idxs] = -W[idxs]
        b[idxs] = -b[idxs]
        # prev_layer_bounds = np.hstack((self.nn.layers[min_layer-1]['Relu_lb'],self.nn.layers[min_layer-1]['Relu_ub']))
        poset = Poset(nn.input_bound,W,b,x)
        poset.build_poset()
        if(len(poset.hashMap) < 2):
            return -1,None
        compute_successors(poset.root)
        successors = [child.num_successors for child in poset.root.children]
        print(successors)
        inverted_relu = poset.root.children[np.argmin(successors)].fixed_faces.pop()
        min_idx = top_k[inverted_relu]
        relu_idx = layer_infeasible[min_idx]
        phase = 1
        if(min_idx in idxs):
            phase = 0
        return relu_idx,phase
        # except Exception as e:
        #     print(e)
        
    def fix_after_propgt(self,model,nn):
        fixed_relus  = [(self._2dabs[layer_idx][relu_idx],1) for layer_idx,relu_idx in nn.active_relus] 
        fixed_relus += [(self._2dabs[layer_idx][relu_idx],0) for layer_idx,relu_idx in nn.inactive_relus] 
        for relu_idx,phase in fixed_relus:
            if(phase == 1 and model.getConstrByName("%d_active"%relu_idx) is None):
                model.addConstr(model.getVarByName(self.slack_vars_names[relu_idx]) == 0,name = "%d_active"%relu_idx)
            elif(phase == 0 and model.getConstrByName("%d_active"%relu_idx) is None):
                model.addConstr(model.getVarByName(self.relu_vars_names[relu_idx]) == 0, name = "%d_inactive"%relu_idx)

    def dfs(self, model, nn, infeasible_relus,fixed_relus,layers_masks, depth = 0,undecided_relus = [],paths = 0):

        #node to be handled
        status = 'UNKNOWN'
        # if(depth>self.MAX_DEPTH):
         #     print("MAX depth")
        #     return status
        relu_idx = None
        # relu_idx,phase = self.pick_one(model, nn, undecided_relus,fixed_relus)
        # print(relu_idx,phase)
        if(relu_idx is None):
            # print('Used orig')
            relu_idx,phase =  infeasible_relus[0]
        nonlin_relus = copy(undecided_relus)
        min_layer,_ = self.abs2d[nonlin_relus[0]]
        #relu_idx,phase = self.split_neuron(infeasible_relus,min_layer*52 +5)
        layer_idx,neuron_idx = self.abs2d[relu_idx]
        if(layer_idx > min_layer):
            relu_idx,phase = nonlin_relus[0],int(model.getVarByName('n[%d]'%nonlin_relus[0]).X >=0)
            layer_idx,neuron_idx = self.abs2d[relu_idx]

        
        nonlin_relus.remove(relu_idx)
        # print('DFS:',depth,"Setting neuron %d to %d"%(relu_idx,phase))
        layers_masks = deepcopy(layers_masks)
        network = deepcopy(nn)
        model1 = model.copy()
        self.set_neuron_bounds(model1,network,layer_idx,neuron_idx,phase,layers_masks)
        fixed_relus.append([relu_idx,phase])
        # s = time()
        # self.__prepare_problem()
        # print('Prep Problem',time() - s)
        self.fix_relu(model1, network, fixed_relus)
        model1.optimize()
        if(model1.Status != 3): #Feasible solution
            self.layer_stats[layer_idx-1][0] +=  1
            SAT,infeasible_set = self.check_SAT(model1)
            valid = self.check_potential_CE(np.array([model1.getVarByName(var_name).X for var_name in self.in_vars_names]).reshape((-1,1)))
            if(SAT or valid):
                #print('Solution found')
                status = 'SolFound'  
            else:
                status = self.dfs(model1, network, infeasible_set,copy(fixed_relus),layers_masks,depth+1,nonlin_relus,paths)
        else:
            self.layer_stats[layer_idx-1][1] += 1
        if(status != 'SolFound'):
            paths[0] += 1 
            # if(self.model.Status == 3):
            #    IIS = self.getIIS('result1.ilp')
            #    if(len(IIS) and relu_idx != IIS[-1] and IIS[-1] in [n_idx for n_idx,_ in fixed_relus]):
            #        self.set_neuron_bounds(layer_idx,neuron_idx,-1,layers_masks)
            #        return status

            model1 = model.copy()
            network = deepcopy(nn)
            phase = 1 - phase
            # print('Backtrack, Setting neuron %d to %d'%(relu_idx,phase))
            self.set_neuron_bounds(model1, network, layer_idx,neuron_idx,phase,layers_masks)
            fixed_relus[-1] = [relu_idx,phase]
            # self.__prepare_problem()
            self.fix_relu(model1,network,fixed_relus)
            model1.optimize()
            if(model1.Status != 3): #Feasible solution
                self.layer_stats[layer_idx-1][0] += 1
                SAT,infeasible_set = self.check_SAT(model1)
                valid = self.check_potential_CE(np.array([model1.getVarByName(var_name).X for var_name in self.in_vars_names]).reshape((-1,1)))
                if(SAT or valid):
                    #print('Solution found')
                    status = 'SolFound'  
                else:
                    status = self.dfs(model1, network, infeasible_set,copy(fixed_relus),layers_masks,depth+1,nonlin_relus,paths)
            else:
                status = 'UNSAT'
                self.layer_stats[layer_idx-1][1] += 1

            #if(status != 'SolFound'):
            #    status = 'UNSAT'
        
            # self.set_neuron_bounds(layer_idx,neuron_idx,-1,layers_masks)
        
        return status
            



    def check_SAT(self,model):   
        y = np.array([model.getVarByName(var_name).X for var_name in self.relu_vars_names[self.__input_dim:]])
        net = np.array([model.getVarByName(var_name).X for var_name in self.net_vars_names[self.__input_dim:]])
        slacks = np.zeros_like(y)
        active_infeas = ((y-net) > eps) * (net > eps) #if y>net in net>0 domain
        inactive_infeas =  ((y > eps) * (net < eps))    #if y > 0 in net<0 domain
        active  = np.sort(np.where(active_infeas == True)[0])
        inactive = np.sort(np.where(inactive_infeas == True)[0])
        slacks[active] = y[active] - net[active]
        slacks[inactive] = y[inactive]
        offset = 0
        infeas_relus=[]
       
        active = list(np.where(active_infeas == True)[0] + self.__input_dim)
        inactive = list(np.where(inactive_infeas == True)[0] + self.__input_dim)            
        infeas_relus = [(n_idx,0) for n_idx in inactive]
        infeas_relus +=  [(n_idx,1) for n_idx in active]
        infeas_relus = sorted(infeas_relus)
        if(len(infeas_relus) is not 0):
            infeas_relus = [(idx,phase) for idx,phase in infeas_relus]
            return False, infeas_relus
        return True, None

    def __prepare_problem(self,model, nn):
        #clear all constraints
        # self.model.remove(self.model.getConstrs())
        #Add external convex constraints
        for constraint in self.linear_constraints:
            vars = [model.getVarByName(var_name) for var_name in constraint['x']]
            model.addConstr(LinExpr(constraint['A'],vars), sense = constraint['sense'], rhs = constraint['rhs'])

        self.__add_NN_constraints(model, nn)
        self.add_objective(model, [])

    

    def add_objective(self,model, fixed_relus = None):
        slacks = [model.getVarByName(var_name) for var_name in self.slack_vars_names[self.__input_dim:]]
        init_weight = 1E-10
        weights = []
        for layer_idx,layer_size in enumerate(self.nn.layers_sizes[1:-1]):
            ub = np.maximum(0,self.nn.layers[layer_idx+1]['in_ub'])
            ub[ub > 0] = 1
            weights += list(init_weight * ub)
            # weights += [1] * layer_size
            init_weight *= 10000

        obj = LinExpr()
        if(fixed_relus):
            for idx in fixed_relus:
                weights[idx - self.__input_dim] = 0

        obj.addTerms(weights,slacks)
        model.setObjective(obj)
        model.update()


        
    
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
