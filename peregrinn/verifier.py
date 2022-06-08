'''
Haitham Khedr

File containing the main code for the NN verifier 
'''
from __future__ import annotations
from collections import defaultdict, deque
from ctypes import c_bool

import multiprocessing as mp
from operator import ne
import queue
import threading
from typing import Tuple 
import torch.nn as nn
import torch
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
    def __init__(self):

        self.task_Q         : mp.Queue[Branch] = mp.Queue()
        self.log_Q          : mp.Queue[logging.LogRecord] = mp.Queue()
        self.poison_pill    : mp.Value = mp.Value(c_bool, False)
        self.n_queued_branches     : mp.Value = mp.Value('i', 1)
        self.verified_branches     : mp.Value = mp.Value('I', 0)
        # self.active_workers        : mp.Value = mp.Value('i', 0)
        self.mutex          : mp.Lock = mp.Lock()
        #Make sure only one process writes to this.
        self.verification_result :VerificationResult = VerificationResult()
        

class WorkerData():
    def __init__(self, idx, verifier : Verifier):
        self.worker_idx = idx
        self.num_verified_branches = 0

        # self.solver      = verifier
        self._dfs_branches = deque([])

class Worker(mp.Process):
    def __init__(self,shared : SharedData, private : WorkerData):
        self.name       = f"Worker_{private.worker_idx}"
        super().__init__(name = self.name)
        self._shared    = shared
        self._private   = private
        self._logger = self._createLogger(self._shared.log_Q)
        self._logger.info(f"Initilaized {self.name}")

    def _createLogger(self, log_Q):
        qh = logging.handlers.QueueHandler(log_Q)
        logger = get_logger(self.name, propagate = False, handlers = [qh])
        logger.setLevel(Setting.LOG_LEVEL)
        return logger

    def run(self):
        self._logger.info(f"{self.name} Started...")

        while True:
            #Should terminate?
            poison = self._shared.poison_pill.value
            if(poison == True):
                break

            try:
                branch  = self._shared.task_Q.get(block= False)
                res = branch.verify()
                if res.status == ResultType.UNKNOWN:
                    #TODO: Split and add to the task Q
                    branch_1, branch_2 = branch.split_neuron()
                    self._shared.task_Q.put(branch_1)
                    self._shared.task_Q.put(branch_2)
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
                    self._shared.mutex.acquire()
                    self._shared.verification_result.status = res.status
                    self._shared.verification_result.ce = res.ce
                    self._shared.mutex.release()
            except queue.Empty as e:
                #That's ok, try again.
                pass
            except Exception as e:
                self._logger.exception(e)
        self._logger.info(f"{self.name} Terminated.")

class Branch:
    '''
    Defines a computational branch used in verification
    '''
    def __init__(self, spec: Specification, fixed_neurons = [], gmodel = None, gvars = None) -> None:
        self.input_bounds = spec.input_bounds
        self.spec = spec
        self.fixed_neurons = [] #list of tuples (layer_idx, neuron_idx, phase)
        self.verification_result = VerificationResult(result=ResultType.UNKNOWN)
        self.gmodel = None
        self.gvars = None
        if fixed_neurons:
            self.fixed_neurons = fixed_neurons
        if gmodel:
            self.gmodel = gmodel
        if gvars:
            self.gvars = gvars
    
    def fix_neuron(self, layer_idx, neuron_idx, phase):
        self.fixed_neurons.append((layer_idx, neuron_idx, phase))

    def verify(self) -> VerificationResult:
        if(self.verification_result.status == ResultType.NO_SOL):
            return VerificationResult(result = ResultType.NO_SOL)
        return VerificationResult()

    def split_neuron(self) -> Tuple[Branch, Branch]:
        b1 = self.clone()
        b2 = self.clone()
        b1.verification_result.status = ResultType.NO_SOL
        b2.verification_result.status = ResultType.NO_SOL
        return (b1,b2)

    def clone(self):
        gmodel_cp = self.gmodel.copy()
        fixed_neurons_cp = self.fixed_neurons.copy()
        new_branch = Branch(self.spec, fixed_neurons_cp, gmodel_cp, self.gvars)
        return new_branch   


class Verifier:
    
    '''
    A class that is used for the Verification of a given spec

    Attributes
    ----------
    model   : nn.Module
        PyTorch model
    spec    : Specification
        the vnnlib specification
    int_net : nn.Module
        Interval network for propagating symbolic bounds through the model


    Methods
    -------
    verify(timeout= Setting.TIMEOUT)
        verifies the NN against spec
    '''

    def __init__(self, model: nn.Module, vnnlib_spec: Specification):

        self.model = model
        self.input_bounds = vnnlib_spec.input_bounds
        self.spec = vnnlib_spec
        self.int_net = IntervalNetwork(self.model, self.input_bounds, 
                            operators_dict=OPS.op_dict, in_shape = self.spec.input_shape).to(Setting.DEVICE)
        self.stable_relus, self.unstable_relus = self._compute_init_bounds(method = 'symbolic')


        #Initialize gurobi model
        self.gmodel, self.gvars = self._init_grb_model()
        
        #Initial computational Branch
        self.init_branch = Branch(self.spec, self.gmodel, self.gvars)
        self.verification_result = VerificationResult(ResultType.UNKNOWN)

    def _compute_init_bounds(self, method = 'symbolic'):
        bounds_timer = time.perf_counter()
        if(method == 'symbolic'):
            I = torch.zeros((self.input_bounds.shape[0], 
                            self.input_bounds.shape[0]+ 1), dtype = Setting.TORCH_PRECISION)
            I = I.fill_diagonal_(1).unsqueeze(0)
            input_bounds = self.input_bounds.unsqueeze(0).unsqueeze(1)
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
                unstable_relus[l_idx] = []
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

                stable_relus[l_idx].extend([(relu_idx, 1) for relu_idx in active_relu_idx])
                stable_relus[l_idx].extend([(relu_idx, 0) for relu_idx in inactive_relu_idx])
                unstable_relus[l_idx].extend([relu_idx for relu_idx in unstable_relu_idx])
        
        logger.debug(f"Initial bounds computation took {time.perf_counter() - bounds_timer:.2f} seconds")
        return stable_relus, unstable_relu_idx


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
                l_var = defaultdict(list)
                for neuron_idx in range(lb.shape[0]): 
                    #two vars per relu
                    in_var = vars[-1]['net'][neuron_idx]
                    rvar = gmodel.addVar(name = f"relu[{l_idx}]", lb  = lb[neuron_idx], ub = ub[neuron_idx])
                    svar = gmodel.addVar(name = f"slack[{l_idx}]", lb  = 0)

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

        gmodel = grb.Model()
        gmodel.setParam('OutputFlag', False)
        gmodel.setParam('Threads', 1)
        self.int_net
        vars = []

        #Create variables
        input_dims = self.input_bounds.shape[0]
        lb = self.input_bounds[:,0]
        ub = self.input_bounds[:,1]
        in_vars = gmodel.addVars(input_dims, name = "inp", lb  = lb, ub = ub)
        vars.append({'net' : in_vars.values()})
        for l_idx, layer in enumerate(self.int_net.layers):
            __grb_init_layer(l_idx, layer)
        
        

        logger.debug(f"Gurobi model init time: {time.perf_counter() - model_init_counter:.2f} seconds")
        return gmodel, vars

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


    def _check_violated_samples(self, objectives : list, outputs : torch.tensor) -> tuple[ResultType,torch.tensor]:
        for A,b in objectives:
            counterexamples = torch.where(outputs @ A.T < torch.from_numpy(b))[0]
            if(len(counterexamples) > 0):
                return (ResultType.SOL_FOUND, outputs[0])

        return (ResultType.UNKNOWN, torch.tensor([]))


    def verify(self, timeout : float = Setting.TIMEOUT) -> VerificationResult:
        #TODO: Set SIGALARM handler
        start = time.perf_counter()
        if Setting.TRY_SAMPLING:
            sampling_timer = time.perf_counter() 
            in_bounds_reshaped = self.input_bounds.reshape(*self.spec.input_shape ,2)
            samples = sample_network(in_bounds_reshaped, Setting.N_SAMPLES).to(Setting.DEVICE)
            outputs = self.model(samples).to('cpu')
            status,ce = self._check_violated_samples(self.spec.objectives, outputs)
            logger.debug(f"Sampling verification time: {time.perf_counter()-sampling_timer:.2f} seconds")
            if(status == ResultType.SOL_FOUND):
                #TODO: print counterexample somwhere?
                self.verification_result.status = status
                self.verification_result.ce = ce
                return self.verification_result

        if Setting.TRY_OVERAPPROX:
            overapprox_timer = time.perf_counter()
            output_bounds = self.int_net.layers[-1].post_symbolic.concrete_bounds.squeeze().to('cpu')
            status, _ = self._check_violated_bounds(self.spec.objectives, output_bounds)
            logger.debug(f"Try overapproximation Verification time: {time.perf_counter()-overapprox_timer:.2f} seconds")
            if(status == ResultType.NO_SOL):
                self.verification_result.status = status
                return self.verification_result
        
        #Init workers
        num_workers = 1 if Setting.N_VERIF_CORES <= 1 else Setting.N_VERIF_CORES
        shared_state = SharedData()

        #Start Monitor thread
        mthread = threading.Thread(target=monitor_thread, args=(shared_state.log_Q,))
        mthread.start()
        
        #Init task
        shared_state.task_Q.put(self.init_branch)
        
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
        
        end = time.perf_counter()
        logger.debug(f"Total Verification time: {end-start:.2f} seconds")


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

