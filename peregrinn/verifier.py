'''
Haitham Khedr

File containing the main code for the NN verifier 
'''
from __future__ import annotations
from collections import OrderedDict, defaultdict, deque
from ctypes import c_bool

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from operator import ne
import queue
import threading
from typing import Tuple 
from intervals.symbolic_interval import SymbolicInterval

from utils.config import Setting
from intervals.interval_network import IntervalNetwork
from operators import registered_ops as OPS
from utils.sample_network import sample_network
from utils.specification import Specification
import logging
import logging.handlers
import time
from enum import Enum
from copy import deepcopy

import gurobipy as grb

from operators.linear import * 
from operators.flatten import *
from operators.activations import *

from utils.Logger import get_logger
logger = get_logger(__name__)

class ResultType(Enum):
    SOL_FOUND = 0
    NO_SOL = 1
    TIMEOUT = 2
    UNKNOWN = 3

class VerificationResult:
    def __init__(self, result = ResultType.UNKNOWN, ce :torch.tensor = torch.tensor([])):
        self.status = result
        self.ce = ce


def monitor_thread(log_Q):
    while True:
        record = log_Q.get()
        if record is None:
            break
        root_logger = logging.getLogger()
        root_logger.handle(record)


class SharedData():
    def __init__(self, in_shape, out_shape):

        self.task_Q         : mp.Queue[Branch] = mp.Queue()
        self.log_Q          : mp.Queue[logging.LogRecord] = mp.Queue()
        self.poison_pill    : mp.Value = mp.Value(c_bool, False)
        self.n_queued_branches     : mp.Value = mp.Value('i', 1)
        self.verified_branches     : mp.Value = mp.Value('I', 0)
        # self.active_workers        : mp.Value = mp.Value('i', 0)
        #Make sure only one process writes to this.
        self.status = mp.Value('i',3)
        self.in_tensor = torch.zeros((in_shape))
        self.in_tensor = self.in_tensor.share_memory_()
        self.out_tensor = torch.zeros((out_shape))
        self.out_tensor = self.out_tensor.share_memory_()
        
    @property
    def counterexample(self):
        return {'input': self.in_tensor, 'output': self.out_tensor}
    @property
    def verification_result(self):
        return VerificationResult(ResultType(self.status.value), self.counterexample)

class WorkerData():
    def __init__(self, idx, verifier : Verifier):
        self.worker_idx = idx
        self.num_verified_branches = 0
        self.verifier      = verifier
        self._dfs_branches = deque([])

class Worker(mp.Process):
    def __init__(self,shared : SharedData, private : WorkerData):
        self.name       = f"Worker_{private.worker_idx}"
        super().__init__(name = self.name)
        self._shared    = shared
        self._private   = private
        self._logger = self._createLogger(self._shared.log_Q)
        self._logger.info(f"Initilaized {self.name}")
        self.verifier = self._private.verifier

    def _createLogger(self, log_Q):
        qh = logging.handlers.QueueHandler(log_Q)
        logger = get_logger(self.name, propagate = False, handlers = [qh])
        logger.setLevel(Setting.LOG_LEVEL)
        return logger

    def run(self):
        torch.set_num_threads(1)
        self._logger.info(f"{self.name} Started...")
        while True:
            #Should terminate?
            poison = self._shared.poison_pill.value
            if(poison == True):
                break

            try:
                if(self._private._dfs_branches):
                    branch = self._private._dfs_branches.popleft()
                else:
                    branch  = self._shared.task_Q.get(block = False)
                    
                sol = branch.verify(self.verifier)
                res = branch.verification_result
                if res.status == ResultType.UNKNOWN:
                    #TODO: Split and add to the task Q
                    branch_1, branch_2 = branch.split_neuron(sol)
                    # self._shared.task_Q.put(branch_1)
                    # self._shared.task_Q.put(branch_2)
                    self._private._dfs_branches.append(branch_1)
                    self._private._dfs_branches.append(branch_2)
                    with self._shared.n_queued_branches.get_lock():
                        self._shared.n_queued_branches.value += 2
                else:
                    with self._shared.verified_branches.get_lock():
                        self._shared.verified_branches.value += 1

                with self._shared.n_queued_branches.get_lock():
                    self._shared.n_queued_branches.value -= 1 #The original branch is removed

                if(self._shared.n_queued_branches.value <= 0 or res.status == ResultType.SOL_FOUND):
                    #Last Branch or unsafe one
                    self._shared.poison_pill.value = True
                    # self._shared.mutex.acquire()
                    self._shared.status.value = res.status.value
                    self._shared.in_tensor[:]  = res.ce['x']
                    self._shared.out_tensor[:] = res.ce['y']
                    # self._shared.mutex.release()
                    if(res.status == ResultType.SOL_FOUND):
                        logger.debug(f"{self.name} found a counterexample")
                    else:
                        logger.debug(f"{self.name} verified branch")
            except queue.Empty as e:
                #That's ok, try again.
                pass
            except Exception as e:
                self._logger.exception(e)
                raise e
        self._logger.info(f"{self.name} Terminated.")

class Branch:
    '''
    Defines a computational branch used in verification
    '''
    def __init__(self, spec: Specification, input_interval : SymbolicInterval, fixed_neurons = {}, curr_layer = 0, unstable_relus = None) -> None:
        self.input_bounds = spec.input_bounds
        self.spec = spec
        self.input_interval = input_interval
        self.fixed_neurons = defaultdict(list) #dict of list tuples layer_idx -> list[(neuron_idx, phase)]
        self.verification_result = VerificationResult(result=ResultType.UNKNOWN)
        self.curr_layer = torch.inf #Current layer where splitting occurs
        self.unstable_relus = None
        
        if fixed_neurons:
            self.fixed_neurons = fixed_neurons
        if curr_layer:
            self.curr_layer = curr_layer
        if unstable_relus:
            self.unstable_relus = unstable_relus
        

    def _fix_neuron(self, layer_idx, neuron_idx, phase):
        #Check if there is unconditioned neurons in current layer
        if((layer_idx > self.curr_layer) and (len(self.unstable_relus[self.curr_layer]) > 0)):
            #condition on any neuron in the curr_layer
            neuron_idx = self.unstable_relus[self.curr_layer].pop()
            layer_idx = self.curr_layer
        else:
            self.curr_layer = layer_idx
            self.unstable_relus[self.curr_layer].remove(neuron_idx)

        self.fixed_neurons[layer_idx].append((neuron_idx, phase))
    def _update_grb_model(self, gmodel, gvars, int_net : IntervalNetwork, unstable_relus, fixed_neurons):

        def _update_grb_layer(l_idx, layer , unstable_relus_idx):

            lb = layer.post_conc_lb.squeeze()
            ub = layer.post_conc_ub.squeeze()
            layer_vars = gvars[l_idx]
            if type(layer) == Linear:
                for idx, v in enumerate(layer_vars['net']):
                    var = gmodel.getVarByName(v.VarName)
                    var.lb = lb[idx].item()
                    var.ub = ub[idx].item()
                
            elif type(layer) == ReLU:
                #Assumes a linear layer before a ReLU
                pre_lb = layer.pre_conc_lb.squeeze()
                pre_ub = layer.pre_conc_ub.squeeze()
                for neuron_idx in unstable_relus_idx[l_idx]: 
                    #two vars per relu
                    rvar = gmodel.getVarByName(layer_vars['net'][neuron_idx].varName)
                    svar = gmodel.getVarByName(layer_vars['slack'][neuron_idx].varName)
                    in_var = gmodel.getVarByName(gvars[l_idx - 1]['net'][neuron_idx].varName)
                    #Active relus
                    if(pre_lb[neuron_idx] > 0):
                        rvar.lb = pre_lb[neuron_idx]
                        rvar.ub = pre_ub[neuron_idx]
                        gmodel.addConstr(svar == 0, name= f"{neuron_idx}_active")
                    elif(pre_ub[neuron_idx] <= 0):
                        rvar.lb = 0
                        rvar.ub = 0
                        gmodel.addConstr(rvar == 0, name= f"{neuron_idx}_inactive") #For completeness
                    else:
                        rvar.ub = pre_ub[neuron_idx]
                        factor = (pre_ub[neuron_idx] / (pre_ub[neuron_idx]-pre_lb[neuron_idx])).item()
                        gmodel.addConstr(rvar <= factor * (in_var- pre_lb[neuron_idx]),name=f"{neuron_idx}_relaxed")
                        A_up = layer.post_symbolic.u.squeeze()[neuron_idx]
                        input_vars = [gmodel.getVarByName(v.varName) for v in gvars[0]['net']]
                        gmodel.addConstr(grb.LinExpr(A_up[:-1],input_vars)  + A_up[-1]  >= rvar,name= f"{neuron_idx}_sym_UB")
                        # gmodel.addConstr(rvar >= in_var) #Not needed since svar = rvar - in_var >=0
                for neuron_idx, phase in fixed_neurons[l_idx]:
                    rvar = gmodel.getVarByName(layer_vars['net'][neuron_idx].varName)
                    svar = gmodel.getVarByName(layer_vars['slack'][neuron_idx].varName)
                    if (phase == 1):
                        if(gmodel.getConstrByName(f"{neuron_idx}_active") is None):
                            rvar.lb = 0
                            rvar.ub = pre_ub[neuron_idx]
                            gmodel.addConstr(svar == 0, name= f"{neuron_idx}_active")
                    else:
                        rvar.lb = 0
                        rvar.ub = 0

            elif type(layer) == Flatten:
                return

            elif type(layer) == Conv2d:
                raise NotImplementedError()

            gmodel.update()


        for l_idx, layer in enumerate(int_net.layers):
            _update_grb_layer(l_idx, layer, unstable_relus)

    def verify(self, verifier : Verifier) -> VerificationResult:
        
        if(self.verification_result.status != ResultType.UNKNOWN):
            return None
        # else:
        #     assert self.verification_result.status == ResultType.UNKNOWN, "Branch is already verified"

        gmodel  = verifier.gmodel
        gvars   = verifier.gvars
        int_net = verifier.int_net
        if(self.curr_layer != torch.inf): 
            
            gmodel  = verifier.gmodel.copy()
            #Propagate bounds
            int_net(self.input_interval, self.fixed_neurons)
            #update solver
            self._update_grb_model(gmodel, gvars, int_net, self.unstable_relus, self.fixed_neurons)
        
        try:
            gmodel.optimize()
            if gmodel.status == grb.GRB.INFEASIBLE:
                self.verification_result.status = ResultType.NO_SOL
            elif gmodel.status == grb.GRB.OPTIMAL:
                # int_net = verifier.int_net
                x = torch.tensor([gmodel.getVarByName(v.varName).X for v in gvars[0]['net']],dtype = Setting.TORCH_PRECISION)
                y = int_net.evaluate(x.to(Setting.DEVICE))
                result = verifier._check_violated_samples(verifier.spec.objectives, y)
                if result.status == ResultType.SOL_FOUND:
                    self.verification_result.status = result.status
                    self.verification_result.ce = {'x':x.reshape(verifier.spec.input_shape),
                                                    'y':y.reshape(verifier.spec.output_shape)}
            else:
                logger.exception(f"Numerical Error solving gurobi model - status: {gmodel.status}")
            return (gmodel, gvars)
        except Exception as e:
            logger.exception(e)
            raise e

    def split_neuron(self, sol: Tuple) -> Tuple[Branch, Branch]:
        l_idx, n_idx, phase = self._pick_neuron(sol)
        b1 = self.clone()
        b2 = self.clone()
        b1._fix_neuron(l_idx, n_idx, phase)
        b2._fix_neuron(l_idx, n_idx, 1 - phase)

        return (b1,b2)

    def clone(self):
        # gmodel_cp = self.gmodel.copy()
        fixed_neurons_cp = deepcopy(self.fixed_neurons)
        unstable_relus = deepcopy(self.unstable_relus)
        new_branch = Branch(self.spec, self.input_interval, fixed_neurons = fixed_neurons_cp
                            , curr_layer= self.curr_layer, unstable_relus = unstable_relus)
        return new_branch   

    def _pick_neuron(self, sol : Tuple):
        layers_slacks = self._get_layer_slacks(sol)
        for k,v in layers_slacks.items():
            if(len(v) > 0):
                l_idx, shallowest_layer_slacks = k,v
                break
        infeasible_relus_idx = shallowest_layer_slacks.nonzero()
        slacks = shallowest_layer_slacks[infeasible_relus_idx]
        #Some logic to pick one neuron
        n_idx = infeasible_relus_idx[0].item()
        phase = int((shallowest_layer_slacks[n_idx] > 0).item())
        

        return (l_idx, n_idx, phase)
   
    def _get_layer_slacks(self, sol: Tuple):
        #Assumes ReLU is never the first layer
        gmodel, gvars = sol
        layers_infeas = {}
        for l_idx, layer_vars in enumerate(gvars):
            if 'slack' in layer_vars.keys():
                #Layer is ReLU 
                #Get pre, post relu and slacks
                net = torch.tensor([gmodel.getVarByName(v.varName).X \
                    for v in gvars[l_idx-1]['net']], dtype = Setting.TORCH_PRECISION)
                y = torch.tensor([gmodel.getVarByName(v.varName).X \
                    for v in gvars[l_idx]['net']], dtype = Setting.TORCH_PRECISION)
                s = torch.tensor([gmodel.getVarByName(v.varName).X \
                    for v in gvars[l_idx]['slack']], dtype = Setting.TORCH_PRECISION)
                s_neg_idx = torch.where (net < 0)[0] 
                s[s_neg_idx] = -y[s_neg_idx]
                infeasible_layer = (torch.sum(s) != 0).item()
                layers_infeas[l_idx] = s if infeasible_layer else torch.tensor([])
                if(infeasible_layer and Setting.ONLY_FIRST_INFEASIBLE_LAYER):
                    break
        return layers_infeas
            


    

class Verifier:
    
    '''
    Main class that is used for the Verification of a given spec

    Attributes
    ----------
    model   : nn.Module
        PyTorch model
    spec    : Specification
        the vnnlib specification (assumes only one objective)
    int_net : nn.Module
        Interval network for propagating symbolic bounds through the model
    objective_idx : int
        Index of the objective (if spec.objectives has multiple objectives)


    Methods
    -------
    verify(timeout= Setting.TIMEOUT)
        verifies the NN against spec
    '''

    def __init__(self, model: nn.Module, vnnlib_spec: Specification, objective_idx = 0):

        self.model = model
        self.model.share_memory()
        self.input_bounds = vnnlib_spec.input_bounds
        self.spec = vnnlib_spec
        self.objective_idx = objective_idx
        self.int_net = IntervalNetwork(self.model, self.input_bounds, 
                            operators_dict=OPS.op_dict, in_shape = self.spec.input_shape).to(Setting.DEVICE)
        self.int_net.share_memory()
        self.stable_relus, self.unstable_relus, self.input_interval = self._compute_init_bounds(method = 'symbolic')


        self.gmodel, self.gvars = self._init_grb_model()
        #Initial computational Branch
        self.verification_result = VerificationResult(ResultType.UNKNOWN)


    def _init_grb_model(self):

        model_init_counter = time.perf_counter()
        def __grb_init_layer(l_idx, layer):
            lb = layer.post_conc_lb.squeeze()
            ub = layer.post_conc_ub.squeeze()
            input_dims = lb.shape[0]
            layer_vars = {}
            if type(layer) == Linear:
                l_vars = gmodel.addVars(input_dims, name = f"lay[{l_idx}]", lb  = lb, ub = ub).values()
                W = layer.torch_layer.weight
                b = layer.torch_layer.bias
                for neuron_idx in range(W.shape[0]):
                    lin_expr =  grb.LinExpr(W[neuron_idx], vars[-1]['net'])
                    gmodel.addConstr(l_vars[neuron_idx] == (lin_expr + b[neuron_idx]))
                layer_vars['net'] = l_vars
                
            elif type(layer) == ReLU:
                #Assumes a linear layer before a ReLU
                pre_lb = layer.pre_conc_lb.squeeze()
                pre_ub = layer.pre_conc_ub.squeeze()
                l_var = {'net':[], 'slack': []}
                for neuron_idx in range(lb.shape[0]): 
                    #two vars per relu
                    in_var = vars[-1]['net'][neuron_idx]
                    rvar = gmodel.addVar(name = f"relu[{l_idx}][{neuron_idx}]", lb  = lb[neuron_idx], ub = ub[neuron_idx])
                    svar = gmodel.addVar(name = f"slack[{l_idx}][{neuron_idx}]", lb  = 0)

                    l_var['net'].append(rvar)
                    l_var['slack'].append(svar)
                    gmodel.addConstr(svar == rvar - in_var)

                    #Active relus
                    if(pre_lb[neuron_idx] > 0):
                        gmodel.addConstr(svar == 0, name= f"{neuron_idx}_active")
                    elif(pre_ub[neuron_idx] <= 0):
                        gmodel.addConstr(rvar == 0, name= f"{neuron_idx}_inactive")
                    else:
                        factor = (pre_ub[neuron_idx] / (pre_ub[neuron_idx]-pre_lb[neuron_idx])).item()
                        gmodel.addConstr(rvar <= factor * (in_var- pre_lb[neuron_idx]),name=f"{neuron_idx}_relaxed")
                        A_up = layer.post_symbolic.u.squeeze()[neuron_idx]
                        gmodel.addConstr(grb.LinExpr(A_up[:-1],vars[0]['net'])  + A_up[-1]  >= rvar,name= f"{neuron_idx}_sym_UB")
                        # gmodel.addConstr(rvar >= in_var) #Not needed since svar = rvar - in_var >=0
                layer_vars = l_var
            elif type(layer) == Flatten:
                return

            elif type(layer) == Conv2d:
                raise NotImplementedError()

            vars.append(layer_vars)
            gmodel.update()

        
        def _grb_init_objective(gmodel, gvars, stable_relus):

            obj = grb.LinExpr()
            init_weight = 1E-10
            for l_idx, layer in enumerate(self.int_net.layers):
                if 'slack' in  gvars[l_idx]:
                    #Layer is ReLU
                    pre_ub = layer.pre_conc_ub.squeeze()
                    slacks = gvars[l_idx]['slack']
                    
                    ub = torch.maximum(torch.zeros_like(pre_ub),pre_ub)
                    ub[ub > 0] = 1
                    weights = list(init_weight * ub)
                    init_weight *= 10000
                    # layer_stable = stable_relus[l_idx]
                    # for neuron_idx,_ in layer_stable:
                    #     weights[neuron_idx] = 0

                    obj.addTerms(weights,slacks)

            gmodel.setObjective(obj)
            gmodel.update()
        
        gmodel = grb.Model()
        gmodel.setParam('OutputFlag', False)
        gmodel.setParam('Threads', 1)
        vars = []

        #Create variables
        input_dims = self.input_bounds.shape[0]
        lb = self.input_bounds[:,0]
        ub = self.input_bounds[:,1]
        in_vars = gmodel.addVars(input_dims, name = "inp", lb  = lb, ub = ub)
        vars.append({'net' : in_vars.values()})
        #Add NN constraints
        for l_idx, layer in enumerate(self.int_net.layers):
            __grb_init_layer(l_idx, layer)

        _grb_init_objective(gmodel, vars, self.stable_relus)

        # Add output constraints
        A,b = self.spec.objectives[self.objective_idx]
        for row, rhs in zip(A,b):
            gmodel.addConstr(row @ vars[-1]['net'] <= rhs) 
        logger.debug(f"Gurobi model init time: {time.perf_counter() - model_init_counter:.2f} seconds")
        return gmodel, vars


    def _compute_init_bounds(self, method = 'symbolic'):
        bounds_timer = time.perf_counter()
        if(method == 'symbolic'):
            I = torch.zeros((self.input_bounds.shape[0], 
                            self.input_bounds.shape[0]+ 1), dtype = Setting.TORCH_PRECISION)
            I = I.fill_diagonal_(1).unsqueeze(0)#.detach()
            input_bounds = self.input_bounds.unsqueeze(0).unsqueeze(1)#.detach()
            layer_sym = SymbolicInterval(input_bounds.to(Setting.DEVICE),I,I, device = Setting.DEVICE)
            layer_sym.concretize()
            self.int_net(layer_sym)
        else:
            raise NotImplementedError()
        
        stable_relus = {}
        unstable_relus= {}
        for l_idx, layer in enumerate(self.int_net.layers):
            if type(layer) == ReLU:
                stable_relus[l_idx] = []
                unstable_relus[l_idx] = set()
                lb = layer.pre_conc_lb.squeeze()
                ub = layer.pre_conc_ub.squeeze()
                active_relu_idx = torch.where(lb > 0)[0]
                inactive_relu_idx = torch.where(ub <= 0)[0]
                unstable_relu_idx = torch.where((ub > 0) * (lb <0))[0]
                try:
                    assert active_relu_idx.shape[0] + inactive_relu_idx.shape[0] \
                            + unstable_relu_idx.shape[0] == lb.shape[0]
                except AssertionError as e:
                    logger.error("Assertion failed(Shape mismatch): Some Relus are neither stable nor unstable")

                stable_relus[l_idx].extend([(relu_idx.item(), 1) for relu_idx in active_relu_idx])
                stable_relus[l_idx].extend([(relu_idx.item(), 0) for relu_idx in inactive_relu_idx])
                unstable_relus[l_idx].update([relu_idx.item() for relu_idx in unstable_relu_idx])
        
        logger.debug(f"Initial bounds computation took {time.perf_counter() - bounds_timer:.2f} seconds")
        return stable_relus, unstable_relus, layer_sym


    def _check_violated_bounds(self, objectives : list, bounds : torch.tensor) -> tuple[ResultType,torch.tensor]:
        for A,b in objectives:
            A = torch.from_numpy(A).type(Setting.TORCH_PRECISION)
            b = torch.from_numpy(b).type(Setting.TORCH_PRECISION)
            zeros = torch.zeros_like(A)
            A_pos = torch.maximum(zeros, A)
            A_neg = torch.minimum(zeros, A)
            lower_bound = A_pos @ bounds[:,0] + A_neg @ bounds[:,1]
            if(lower_bound > b):
                return (ResultType.NO_SOL, torch.tensor([]))
            else:
                return (ResultType.UNKNOWN, torch.tensor([]))


    def _check_violated_samples(self, objectives : list, outputs : torch.tensor) -> VerificationResult:
        result = VerificationResult()
        for A,b in objectives:
            counterexamples = torch.where(outputs @ A.T < torch.from_numpy(b))[0]
            if(len(counterexamples) > 0):
                result.status = ResultType.SOL_FOUND
                result.ce = counterexamples[0]
                return result

        return result


    def verify(self, timeout : float = Setting.TIMEOUT) -> VerificationResult:
        #TODO: Set SIGALARM handler
        start = time.perf_counter()
        if Setting.TRY_SAMPLING:
            sampling_timer = time.perf_counter() 
            in_bounds_reshaped = self.input_bounds.reshape(*self.spec.input_shape ,2)
            samples = sample_network(in_bounds_reshaped, Setting.N_SAMPLES).to(Setting.DEVICE)
            outputs = self.model(samples).to('cpu')
            result = self._check_violated_samples(self.spec.objectives, outputs)
            logger.debug(f"Sampling verification time: {time.perf_counter()-sampling_timer:.2f} seconds")
            if(result.status == ResultType.SOL_FOUND):
                logger.info("Found counterexample by sampling")
                #TODO: print counterexample somwhere?
                self.verification_result.status = result.status
                self.verification_result.ce = {'x': samples[result.ce],'y': outputs[result.ce]}
                return self.verification_result

        if Setting.TRY_OVERAPPROX:
            overapprox_timer = time.perf_counter()
            output_bounds = self.int_net.layers[-1].post_symbolic.concrete_bounds.squeeze().to('cpu')
            status, _ = self._check_violated_bounds(self.spec.objectives, output_bounds)
            logger.debug(f"Try overapproximation Verification time: {time.perf_counter()-overapprox_timer:.2f} seconds")
            if(status == ResultType.NO_SOL):
                logger.info("Property holds with overapproximation")
                self.verification_result.status = status
                return self.verification_result

        #Init Branch
        init_branch = Branch(self.spec, self.input_interval, unstable_relus = self.unstable_relus)

        num_workers = 1 if Setting.N_VERIF_CORES <= 1 else Setting.N_VERIF_CORES
        shared_state = SharedData(self.spec.input_shape, self.spec.output_shape)

        #Start Monitor thread
        mthread = threading.Thread(target=monitor_thread, args=(shared_state.log_Q,))
        mthread.start()
        
        #Init task
        shared_state.task_Q.put(init_branch)
        
        #Start all the workers
        workers_timer = time.perf_counter()
        workers = []
        for i in range(num_workers):
            private_state = WorkerData(i, self)
            workers.append(Worker(shared_state, private_state))
            workers[-1].start()
        logger.debug(f"Creating and starting workers took {time.perf_counter() - workers_timer:.2f} seconds")
        
        for worker in workers:
            worker.join()
        #Terminate monitor thread
        shared_state.log_Q.put(None)
        mthread.join()

        result = shared_state.verification_result
        self.verification_result = result

        end = time.perf_counter()
        logger.debug(f"Total Verification time: {end-start:.2f} seconds")
        return self.verification_result


    @property
    def num_stable_relus(self):
        num_stable_relus = 0
        for layer_stable_relus in self.stable_relus.values():
            num_stable_relus += len(layer_stable_relus)
        return num_stable_relus
    
    @property
    def num_unstable_relus(self):
        num_unstable_relus = 0
        for layer_unstable_relus in self.unstable_relus.values():
            num_unstable_relus += len(layer_unstable_relus)
        return num_unstable_relus

