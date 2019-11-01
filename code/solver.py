from pulp import *
from random import random,seed
from time import time
sys.path.append('./z3/z3-4.4.1-x64-osx-10.11/bin/')
import z3 as z3
import pickle
from Workspace import Workspace
import math
from gurobipy import * 
import sys
import os
eps = 1E-3
class Solver():

    def __init__(self,input_size=16,hidden_units=600,output_dim =2, network = None, maxIter = 100000, process_num = 0, print_lock = None):
        self.maxNumberOfIterations = maxIter
        self.network        = network

        #TODO: self.__parse_network() #compute the dims of input and hidden nodes
        self.__input_dim    = input_size
        self.__hidden_units = hidden_units
        self.__output_dim   = output_dim

        self.SATsolver      = z3.Solver()
        self.convIFClauses  = z3.BoolVector('bConv', self.__hidden_units)
        self.SATsolver.reset()

        #Add variables
        self.state_vars         = LpVariable.dicts("x", [i for i in range(4)])
        self.im_vars            = LpVariable.dicts("i", [i for i in range(self.__input_dim)])
        self.net_vars           = LpVariable.dicts("y", [i for i in range(self.__hidden_units)])
        self.relu_vars          = LpVariable.dicts("z", [i for i in range(self.__hidden_units)])
        self.slack_vars         = LpVariable.dicts("s", [i for i in range(self.__hidden_units)], lowBound = 0)
        self.out_vars           = LpVariable.dicts("u", [i for i in range( self.__output_dim)])
        self.next_state_vars    = LpVariable.dicts("w", [i for i in range(4)])

        #network weights
        seed(12)
        hidden_weights= network[0]
        output_weights= network[1]
        self.W = hidden_weights[0].T
        self.b = hidden_weights[1] 
        self.W_out = output_weights[0].T
        self.b_out = output_weights[1] 
        # self.W  = [[2*(random()-0.5)  for i in range(input_size)] for j in range(hidden_units)]
        # self.W_out = [[2*(random() -0.5 )  for i in range(hidden_units)] for j in range(output_dim)]
        # self.b = [2*(random()-0.5)]* hidden_units
        # self.b_out  = [2*(random()-0.5)]* output_dim

        self.linear_constraints = []
        self.curr_problem = []
        self.preprocessing = True # set to False after first call to solve
        self.preprocess_counter_examples = []
        self.backend = 'Gurobi'

        self.process_num = process_num
        self.print_lock = print_lock

    def add_variables(self, name, num, lowB = None, uppB = None):
        return LpVariable.dicts(name, [i for i in range( num)],lowB, uppB)

    def add_linear_constraints(self, A, x, b, sense = LpConstraintLE):
        for row in range(len(b)):
            linear_expression = LpAffineExpression([(x[col],A[row][col]) for col in range(len(x))])
            constraint =  LpConstraint(linear_expression, sense = sense, rhs = b[row])
            self.linear_constraints.append(constraint)

    def __add_objective_fn(self, problem):
        slack_idx = [i for i in range(self.__hidden_units) if i not in self.preprocess_counter_examples]
        for i in self.preprocess_counter_examples:
            problem += (self.slack_vars[i] == 0)
        problem +=lpSum([(self.slack_vars[i],1) for i in slack_idx])

    def __add_NN_constraints(self, problem, relu_assignment,neurons_indices = None):

        if(neurons_indices is None):
            neurons_indices = range(self.__hidden_units)

        for neuron_idx in neurons_indices:
            #add net constraints
            net_expr = LpAffineExpression([(self.im_vars[input_idx],self.W[neuron_idx][input_idx]) for input_idx in range(self.__input_dim)])
            problem += self.net_vars[neuron_idx] == (net_expr + self.b[neuron_idx])
            
            #add relu inequality constraint
            problem += (1-2*relu_assignment[neuron_idx]) * self.net_vars[neuron_idx] - self.slack_vars[neuron_idx] <=0
           
            problem += self.relu_vars[neuron_idx] == relu_assignment[neuron_idx] * (self.net_vars[neuron_idx] + self.slack_vars[neuron_idx])
        
        if(not self.preprocessing):
            for neuron_idx in range(self.__output_dim):
                net_expr = LpAffineExpression([(self.relu_vars[input_idx],self.W_out[neuron_idx][input_idx]) for input_idx in range(self.__hidden_units)])
                problem += ( self.out_vars[neuron_idx] == (net_expr + self.b_out[neuron_idx]) )
    
    def __extractSATModel(self):
        z3Model                 = self.SATsolver.model()
        convIFModel             = [z3.is_true(z3Model[bConv])   for bConv   in self.convIFClauses]
        return convIFModel

    def solve(self, bool_model = None):
        
        status = 'TLE'
        AND_bool_constraints = True
        if(bool_model is None):
            bool_model = [False] *self.__hidden_units
        if(self.preprocessing):
            counter_example, convIFModel = self.preprocess()
            for idx, neuron_idx in enumerate(counter_example):
                bool_model[neuron_idx] = convIFModel[idx]

            self.preprocessing = False
            # self.__add_counter_example(counter_example, convIFModel, AND_bool_constraints)
            self.preprocess_counter_examples += counter_example
            return self.preprocess_counter_examples, bool_model

        if(self.preprocessing is False):
            AND_bool_constraints = False
        solutionFound = False
        iterationsCounter = -1
        counter_examples = []
        while solutionFound == False and iterationsCounter < self.maxNumberOfIterations:
            iterationsCounter               = iterationsCounter + 1

            if iterationsCounter % 100 == 0:
                self.print_lock.acquire()
                print('******** Solver , iteration = ', iterationsCounter, ', process',self.process_num,'********')
                self.print_lock.release()

            SATcheck    = self.SATsolver.check()
            if  SATcheck == z3.unsat:
                self.print_lock.acquire()
                print('==========  Problem is UNSAT ==========', ', process',self.process_num)
                self.print_lock.release()
                status = 'UNSAT'
                break
            else: #Generate new boolean model
                convIFModel         = self.__extractSATModel()
                # convIFModel = [False, False, True, True, False, False, False, False, True, True, False, True, False, False, True, False, True, False, False, True, False, True, False, False, True, True, False, True, False, True, True, False, True, True, True, True, True, True, False, False, True, False, True, False, True, True, False, False, True, False, True, True, False, True, True, False, True, False, True, False, False, False, True, False, True, False, False, False, False, False, True, False, False, False, True, True, True, True, False, False, True, False, False, False, True, False, False, False, True, False, True, True, True, False, False, True, False, True, False, True, False, True, True, True, False, True, False, False, True, True, False, False, True, False, False, False, False, False, False, False, True, False, True, False, False, True, False, True, True, False, True, True, True, True, True, False, False, False, True, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, True, True, False, False, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, True, False, False, True, True, False, True, False, False, False, False, True, True, True, False, False, False, False, False, False, True, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False, False, True, False, True, False, False, False, False, False, True, True, True, True, True, False, True, False, False, False, False, True, False, False, True, True, True, True, True, False, False, True, True, False, False, False, False, False, False, True, False, True, True, False, True, False, False, False, True, False, False, True, True, False, False, True, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, True, False, False, True, False, True, True, False, False, False, True, True, True, True, False, False, False, False, True, False, True, False, False, False, False, False, False, True, False, True, False, False, False, True, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, True, True, True, True, False, False, False, False, False, True, True, False, True, False, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, True, True, False, False, False, False, False, True, False, True, False, True, False, True, True, False, True, False, False, False, True, False, True, True, False, True, True, False, False, False, False, False, True, False, False, False, True, False, True, True, False, True, False, False, True, True, False, True, False, False, False, False, False, True, True, True, False, True, False, False, True, False, True, True, False, True, False, False, False, False, True, False, False, False, False, False, True, False, False, True, False, True, True, True, False, True, True, False, True, False, False, True, True, True, False, True, False, False, False, True, False, False, True, False, False, False, True, False, False, True, True, False, False, False, True, True, True, True, False, False, True, False, False, True, False, False, False, False, False, False, True, False, False, True, True, False, False, True, False, False, False, False, True, False, True, False, True, True, False, True, True, True, True, False, False, False, False, True, False, False, False, False, False, False, True, True, True, False, False, False, True, False, True, False, False, False, False, True, True, True, True, False, False, True, True, True, True, True, False, True, False, True, True, True, True, False]
                # print 'ConvIfModel = ', [i for i, x in enumerate(convIFModel) if x == True], '\n'
            #prepare problem

            if(self.backend == 'Gurobi'):
                problem = self.__prepare_problem(convIFModel, feasibility = True)
                fname = str(self.process_num)+'_problem.lp'
                problem.writeLP(fname)
                sys.stdout = open(os.devnull, "w")
                model = read(fname)
                os.remove(fname)
                sys.stdout = sys.__stdout__
                # model.params.IISMethod = 1
                # model.params.FeasibilityTol = 1E-2
                model.params.OutputFlag = 0
                model.optimize()
                if(model.Status == 3): #Infeasible
                    IIS_slack = []
                    try:
                        model.computeIIS() 
                        fname = str(self.process_num)+'_result.ilp'
                        model.write(fname)
                        with open(fname,'rb') as f:
                            contents = f.readlines()
                        os.remove(fname)
                        for i in range(len(contents)):
                            idx = len(contents) - 1 -i
                            line = contents[idx]
                            if(line == 'Bounds\n'):
                                break
                            if('s_' in line):
                                words = line.split(' ')[1]
                                words = words.split('_')
                                IIS_slack.append(int(words[1]))

                        if(len(IIS_slack) != 0):
                            self.__add_counter_example(IIS_slack, convIFModel, AND = False)
                            # print(IIS_slack)
                        else:
                            status = "Infeasible"
                            break
                            
                    except Exception as e:
                        self.print_lock.acquire()
                        print(e,self.process_num)
                        self.print_lock.release()
                        problem = self.__prepare_problem(convIFModel, feasibility = False)
                        problem.solve()
                        solver_status = problem.status
                        # print('Switching to Trivial')
                        # self.__add_counter_example(counter_example, convIFModel)
                        if(solver_status == LpStatusOptimal):
                            counter_example = self.__generate_counter_example()
                            if(len(counter_example) == 0):
                                solutionFound = True
                                status = 'solFound'
                                break
                            else:
                                self.__add_counter_example(counter_example, convIFModel, AND = False)
                                self.print_lock.acquire()
                                print('Switching to Trivial CE',self.process_num)
                                self.print_lock.release()
                                # status = 'EXCEPTION'
                else:
                    solutionFound = True
                    status = 'SolFound'      
                    problem = self.__prepare_problem(convIFModel, feasibility = True)
                    problem.solve()
                    if(problem.status != LpStatusOptimal):
                        self.print_lock.acquire()
                        print("Something went wrong :)")
                        self.print_lock.release()

                        status = 'EXCEPTION'
                        break
                    self.print_lock.acquire()
                    print('Solution found')
                    print('x',[self.state_vars[i].varValue for i in range(len(self.state_vars))])
                    print('w',[self.next_state_vars[i].varValue for i in range(len(self.next_state_vars))])
                    print('u',[self.out_vars[i].varValue for i in range(len(self.out_vars))])
                    print('i',[self.im_vars[i].varValue for i in range(len(self.im_vars))])
                    # print('slack',[self.slack_vars[i].varValue for i in range(len(self.slack_vars))])
                    print('Relu',[i for i, x in enumerate(convIFModel) if x == True])
                    self.print_lock.release()
            else:
                problem = self.__prepare_problem(convIFModel)
                self.curr_problem = problem #for debug
                #Solve
                problem.solve()
                solver_status = problem.status
                if(solver_status == LpStatusOptimal):
                    counter_example = self.__generate_counter_example()
                    print(counter_example)
                    if(len(counter_example) == 0):
                        solutionFound = True
                        print('Solution found')
                        print('x',[self.state_vars[i].varValue for i in range(len(self.state_vars))])
                        print('w',[self.next_state_vars[i].varValue for i in range(len(self.next_state_vars))])
                        print('u',[self.out_vars[i].varValue for i in range(len(self.out_vars))])
                        print('i',[self.im_vars[i].varValue for i in range(len(self.im_vars))])
                        # print('slack',[self.slack_vars[i].varValue for i in range(len(self.slack_vars))])
                        print('Relu',[i for i, x in enumerate(convIFModel) if x == True])
                        status = 'SolFound'
                    else:
                        counter_examples.append(counter_example)
                        self.__add_counter_example(counter_example, convIFModel, AND_bool_constraints)
                        print('length of counter examples' ,len(counter_example))
                        if(self.preprocessing):
                            self.preprocess_counter_examples = counter_example
                elif(solver_status == LpStatusInfeasible):
                    print("Problem is infeasible")
                    status = 'Infeasible'
                    break
                else:
                    print("Solver error, ERR_CODE:",solver_status)
                    break
                # print('x',[x.varValue for _,x in self.in_vars.items()])
                # print('slack',[slack.varValue for _,slack in self.slack_vars.items()])
                # print(problem.status)
                # print(self.problem)
        self.preprocessing = False
        return problem.variables,counter_examples,status

    def __prepare_problem(self, relu_assignment, feasibility = False):
        problem        = LpProblem("ReluVerify", LpMinimize) 
        #Add external convex constraints
        for constraint in self.linear_constraints:
            problem += constraint
        #Add NN constrainst for unpreprocessed neurons
        # neurons_idx = [idx for idx in range(self.__hidden_units) if idx not in self.preprocess_counter_examples]
        self.__add_NN_constraints(problem, relu_assignment)
        if(feasibility):
            for i in range(self.__hidden_units):
                problem += (self.slack_vars[i] <= eps)
                problem += (self.slack_vars[i] >= -1*eps)
        else:
            self.__add_objective_fn(problem)   

        return problem

    def preprocess(self):
        #Solve the problem for each neuron on its own
        counter_example =[]
        relu_assignment = [0] * self.__hidden_units
        false_assignment = []
        unprocessed_neurons = [idx for idx in range(self.__hidden_units) if idx not in self.preprocess_counter_examples]
        for neuron_idx in unprocessed_neurons:
            for binary_assignment in range(2):
                problem        = LpProblem("ReluVerify", LpMinimize) 
                #Add external convex constraints
                for constraint in self.linear_constraints:
                    problem += constraint
                relu_assignment[neuron_idx] = binary_assignment
                self.__add_NN_constraints(problem,relu_assignment, [neuron_idx])
                self.__add_objective_fn(problem)
                problem.solve()
                if(self.slack_vars[neuron_idx].varValue > eps):
                    counter_example.append(neuron_idx)
                    false_assignment.append(bool(relu_assignment[neuron_idx]))
                    break
        a = [i for idx,i in enumerate(counter_example) if false_assignment[idx] is False]
        # print(a)
        return counter_example, false_assignment



    def __generate_counter_example(self):
        counter_example = []
        for idx,v in self.slack_vars.items():
            if(idx in self.preprocess_counter_examples):
                continue
            if(v.varValue > eps):
                counter_example.append(idx)
        return counter_example

    def add_counter_example(self,counter_example, convIFClause, AND = False):
        self.__add_counter_example(counter_example,convIFClause, AND)
        self.preprocess_counter_examples += counter_example

    def __add_counter_example(self, counter_example, convIFClause,AND = False):
        if(AND): 
            constraint  = z3.And([ self.convIFClauses[idx] !=  convIFClause[idx] for idx in counter_example ])
        else:
            constraint  = z3.Or([ self.convIFClauses[idx] !=  convIFClause[idx] for idx in counter_example ])

        self.SATsolver.add(constraint)
