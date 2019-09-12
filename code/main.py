from pulp import *
from random import random,seed
from time import time
sys.path.append('./z3/z3-4.4.1-x64-osx-10.11/bin/')
import z3 as z3


class ReluVerify():

    def __init__(self,network = None, maxIter = 10000):

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
        self.in_vars     = LpVariable.dicts("x", [i for i in range( self.__input_dim)], -2.0, 2.0)
        self.net_vars    = LpVariable.dicts("y", [i for i in range(self.__hidden_units)])
        self.relu_vars   = LpVariable.dicts("z", [i for i in range(self.__hidden_units)])
        self.slack_vars  = LpVariable.dicts("s", [i for i in range(self.__hidden_units)], lowBound = 0)
        self.out_vars    = LpVariable.dicts("o", [i for i in range( self.__output_dim)],0.0, 2.0)

        
    def __add_objective_fn(self, problem):
        problem +=lpSum([(self.slack_vars[i],1) for i in range(self.__hidden_units)])

    def __add_NN_constraints(self, problem, relu_assignment,add_out_constraint = False):

        for neuron_idx in range(self.__hidden_units):
            #add net constraints
            net_expr = LpAffineExpression([(self.in_vars[input_idx],W[neuron_idx][input_idx]) for input_idx in range(self.__input_dim)])
            problem += self.net_vars[neuron_idx] == (net_expr + b[neuron_idx])
            
            #add relu inequality constraint
            problem += (1-2*relu_assignment[neuron_idx]) * self.net_vars[neuron_idx] - self.slack_vars[neuron_idx] <=0
            
            #add relu equality constraint
            problem += self.relu_vars[neuron_idx] == relu_assignment[neuron_idx] * (self.net_vars[neuron_idx] + self.slack_vars[neuron_idx])
   
        if(add_out_constraint):

            for neuron_idx in range(self.__output_dim):
                net_expr = LpAffineExpression([(self.relu_vars[input_idx],W_out[neuron_idx][input_idx]) for input_idx in range(self.__hidden_units)])
                problem += self.out_vars[neuron_idx] == (net_expr + b_out[neuron_idx])

    def __extractSATModel(self):
        z3Model                 = self.SATsolver.model()
        convIFModel             = [z3.is_true(z3Model[bConv])   for bConv   in self.convIFClauses]
        return convIFModel

    def solve(self):

        solutionFound = False
        iterationsCounter = 0
        out_constraints_flag = False
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
                out_constraints_flag = True
                AND_bool_constraints = False
                
            problem = self.__prepare_problem(convIFModel, out_constraints_flag)
            #Solve
            problem.solve()
            solver_status = problem.status
            if(solver_status == LpStatusOptimal):
                counter_example = self.__generate_counter_example()
                if(len(counter_example) == 0):
                    solutionFound = True
                    print('Solution found')
                    print('x',[x.varValue for _,x in self.in_vars.items()])
                    print('o',[o.varValue for _,o in self.out_vars.items()])
                    print('slack',[slack.varValue for _,slack in self.slack_vars.items()])

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
        self.__add_NN_constraints(problem, relu_assignment, out_constraints)
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
    
#Constants
input_size = 16
hidden_units = 500
output_dim = 2

#network weights
seed(12)
W  = [[2*(random()-0.5)  for i in range(input_size)] for j in range(hidden_units)]
W_out = [[2*(random() -0.5 )  for i in range(hidden_units)] for j in range(output_dim)]
b = [2*(random()-0.5)]* hidden_units
b_out  = [2*(random()-0.5)]* output_dim
#SAT solver
# SATsolver = z3.Solver()
bin = [0,1]
assignment = [(x,y,z,w) for x in bin for y in bin for z in bin for w in bin ]

#Problem vars
# problem     = LpProblem("ReluVerify", LpMinimize)
# in_vars     = LpVariable.dicts("x", [i for i in range(input_size)], -2, 2)
# net_vars    = LpVariable.dicts("y", [i for i in range(hidden_units)])
# relu_vars   = LpVariable.dicts("z", [i for i in range(hidden_units)])
# slack_vars  = LpVariable.dicts("s", [i for i in range(hidden_units)], lowBound = 0)

# relu_assignment = assignment[6]

#Objective
# problem +=lpSum([(slack_vars[i],1) for i in range(hidden_units)])

# #Input constraint
# add_input_constraint()

# #Neural network constraints
# add_NN_constraints(problem)

if __name__ == "__main__":
    solver = ReluVerify()
    s = time()
    solver.solve()
    e = time()  
    print('time', e-s)

