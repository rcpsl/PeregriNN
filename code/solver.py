from pulp import *
from random import random,seed
from time import time
sys.path.append('./z3/z3-4.4.1-x64-osx-10.11/bin/')
import z3 as z3
import pickle
from Workspace import Workspace
import math

class Solver():

    def __init__(self,input_size,hidden_units,output_dim, network = None, maxIter = 10000):

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
        self.im_vars            = LpVariable.dicts("i", [i for i in range( self.__input_dim)])
        self.net_vars           = LpVariable.dicts("y", [i for i in range(self.__hidden_units)])
        self.relu_vars          = LpVariable.dicts("z", [i for i in range(self.__hidden_units)])
        self.slack_vars         = LpVariable.dicts("s", [i for i in range(self.__hidden_units)], lowBound = 0)
        self.out_vars           = LpVariable.dicts("u", [i for i in range( self.__output_dim)])
        self.next_state_vars    = LpVariable.dicts("w", [i for i in range(4)])

        #network weights
        seed(12)
        self.W  = [[2*(random()-0.5)  for i in range(input_size)] for j in range(hidden_units)]
        self.W_out = [[2*(random() -0.5 )  for i in range(hidden_units)] for j in range(output_dim)]
        self.b = [2*(random()-0.5)]* hidden_units
        self.b_out  = [2*(random()-0.5)]* output_dim

        self.linear_constraints = []
        self.curr_problem = []


    def add_variables(self, name, num, lowB = None, uppB = None):
        return LpVariable.dicts(name, [i for i in range( num)],lowB, uppB)

    def add_linear_constraints(self, A, x, b, sense = LpConstraintLE):
        for row in range(len(b)):
            linear_expression = LpAffineExpression([(x[col],A[row][col]) for col in range(len(x))])
            constraint =  LpConstraint(linear_expression, sense = sense, rhs = b[row])
            self.linear_constraints.append(constraint)

    def __add_objective_fn(self, problem):
        problem +=lpSum([(self.slack_vars[i],1) for i in range(self.__hidden_units)])

    def __add_NN_constraints(self, problem, relu_assignment):

        for neuron_idx in range(self.__hidden_units):
            #add net constraints
            net_expr = LpAffineExpression([(self.im_vars[input_idx],self.W[neuron_idx][input_idx]) for input_idx in range(self.__input_dim)])
            problem += self.net_vars[neuron_idx] == (net_expr + self.b[neuron_idx])
            
            #add relu inequality constraint
            # problem += (1-2*relu_assignment[neuron_idx]) * self.net_vars[neuron_idx] - self.slack_vars[neuron_idx] <=0
            relu_expr = LpAffineExpression([(self.net_vars[neuron_idx],(1-2*relu_assignment[neuron_idx])),
                                            (self.slack_vars[neuron_idx],-1)       ] )
            relu_constraint1 = LpConstraint(relu_expr,sense = LpConstraintLE,name="_R1_"+str(neuron_idx), rhs= 0)
            problem += relu_constraint1 
            
            #add relu equality constraint
            relu_expr = LpAffineExpression([(self.net_vars[neuron_idx],relu_assignment[neuron_idx]),
                                            (self.slack_vars[neuron_idx], relu_assignment[neuron_idx]),
                                            (self.relu_vars[neuron_idx],-1)])

            relu_constraint2 = LpConstraint(relu_expr,sense = LpConstraintEQ,name="_R2_"+str(neuron_idx), rhs= 0)

            problem += relu_constraint2
            # problem += self.relu_vars[neuron_idx] == relu_assignment[neuron_idx] * (self.net_vars[neuron_idx] + self.slack_vars[neuron_idx])
   
        for neuron_idx in range(self.__output_dim):
            net_expr = LpAffineExpression([(self.relu_vars[input_idx],self.W_out[neuron_idx][input_idx]) for input_idx in range(self.__hidden_units)])
            problem += self.out_vars[neuron_idx] == (net_expr + self.b_out[neuron_idx])
    
    def __extractSATModel(self):
        z3Model                 = self.SATsolver.model()
        convIFModel             = [z3.is_true(z3Model[bConv])   for bConv   in self.convIFClauses]
        return convIFModel

    def solve(self):

        solutionFound = False
        iterationsCounter = 0
        AND_bool_constraints = True
        while solutionFound == False and iterationsCounter < self.maxNumberOfIterations:
            iterationsCounter               = iterationsCounter + 1

            if iterationsCounter % 1 == 0:
                print '******** Solver , iteration = ', iterationsCounter, '********'

            SATcheck    = self.SATsolver.check()
            if  SATcheck == z3.unsat:
                print '==========  Problem is UNSAT =========='
                return 0
            else: #Generate new boolean model
                convIFModel         = self.__extractSATModel()
                # print 'ConvIfModel = ', [i for i, x in enumerate(convIFModel) if x == True], '\n'
            #prepare problem
            if(iterationsCounter>1):
                AND_bool_constraints = False

            problem = self.__prepare_problem(convIFModel)
            self.curr_problem = problem #for debug
            #Solve
            problem.solve()
            solver_status = problem.status
            if(solver_status == LpStatusOptimal):
                counter_example = self.__generate_counter_example()
                if(len(counter_example) == 0 and iterationsCounter > 1):

                    solutionFound = True
                    print('Solution found')
                    print('x',[self.state_vars[i].varValue for i in range(len(self.state_vars))])
                    print('w',[self.next_state_vars[i].varValue for i in range(len(self.next_state_vars))])
                    print('u',[self.out_vars[i].varValue for i in range(len(self.out_vars))])
                    print('i',[self.im_vars[i].varValue for i in range(len(self.im_vars))])
                    # print('slack',[self.slack_vars[i].varValue for i in range(len(self.slack_vars))])
                    return problem.variables
                else:
                    print(counter_example)
                    self.__add_counter_example(counter_example, convIFModel, AND_bool_constraints)
                    print('length of counter examples' ,len(counter_example))
            
            elif(solver_status == LpStatusInfeasible):
                print("Problem is infeasible")
                return
            else:
                print("Solver error, ERR_CODE:",solver_status)

            # print('x',[x.varValue for _,x in self.in_vars.items()])
            # print('slack',[slack.varValue for _,slack in self.slack_vars.items()])
            # print(problem.status)
            # print(self.problem)

    def __prepare_problem(self, relu_assignment, out_constraints = False):
        problem        = LpProblem("ReluVerify", LpMinimize) 
        #Add external convex constraints
        for constraint in self.linear_constraints:
            problem += constraint
        self.__add_NN_constraints(problem, relu_assignment)
        self.__add_objective_fn(problem)
        return problem

    def __generate_counter_example(self):
        counter_example = []
        for idx,v in self.slack_vars.items():
            if(v.varValue > 0):
                counter_example.append(idx)
        return counter_example
    def __add_counter_example(self, counter_example, convIFClause,AND = False):
        if(AND): 
            constraint  = z3.And([ self.convIFClauses[idx] !=  convIFClause[idx] for idx in counter_example ])
        else:
            constraint  = z3.Or([ self.convIFClauses[idx] !=  convIFClause[idx] for idx in counter_example ])

        self.SATsolver.add(constraint)
