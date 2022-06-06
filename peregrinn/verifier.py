'''
Haitham Khedr

File containing the main code for the NN verifier 
'''
from __future__ import annotations
from ctypes import c_bool

import multiprocessing as mp
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
        self.n_active_branches     : mp.Value = mp.Value('i', 1)
        self.verified_branches     : mp.Value = mp.Value('I', 0)
        self.mutex          : mp.Lock = mp.Lock()
        #Make sure only one process writes to this.
        self.verification_result :VerificationResult = VerificationResult()
        

class WorkerData():
    def __init__(self, idx, model : nn.Module, interval_net : IntervalNetwork):
        self.worker_idx = idx
        self.num_verified_branches = 0

        self.model      = model
        self.int_net    = interval_net

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
                    with self._shared.n_active_branches.get_lock():
                        self._shared.n_active_branches.value += 2
                else:
                    with self._shared.verified_branches.get_lock():
                        self._shared.verified_branches.value += 1

                with self._shared.n_active_branches.get_lock():
                    self._shared.n_active_branches.value -= 1 #The original branch is removed

                if(self._shared.n_active_branches.value <= 0 or res.status == ResultType.SOL_FOUND):
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
    def __init__(self, spec: Specification) -> None:
        self.input_bounds = spec.input_bounds
        self.spec = spec
        self.fixed_neurons = [] #list of tuples (layer_idx, neuron_idx, phase)
        self.verification_result = VerificationResult(result=ResultType.UNKNOWN)

    def fix_neuron(self, layer_idx, neuron_idx, phase):
        self.fixed_neurons.append((layer_idx, neuron_idx, phase))

    def verify(self) -> VerificationResult:
        if(self.verification_result.status == ResultType.NO_SOL):
            return VerificationResult(result = ResultType.NO_SOL)
        return VerificationResult()

    def split_neuron(self) -> Tuple[Branch, Branch]:
        b1 = Branch(self.spec)
        b1.verification_result = VerificationResult(result= ResultType.NO_SOL)
        b2 = b1.clone()
        b2.verification_result = VerificationResult(result= ResultType.NO_SOL)
        return (b1,b2)

    def clone(self):
        new_branch = Branch(self.spec)
        new_branch.fixed_neurons = self.fixed_neurons.copy()
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
    '''

    def __init__(self, model: nn.Module, vnnlib_spec: Specification):

        self.model = model
        self.input_bounds = vnnlib_spec.input_bounds
        self.spec = vnnlib_spec
        self.int_net = IntervalNetwork(self.model, self.input_bounds, 
                            operators_dict=OPS.op_dict, in_shape = self.spec.input_shape).to(Setting.DEVICE)
        
        self.init_branch = Branch(self.spec)
        self.verification_result = VerificationResult(ResultType.UNKNOWN)

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

    def verify(self, timeout : float = Setting.TIMEOUT) -> ResultType:
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
                return status

        if Setting.TRY_OVERAPPROX:
            overapprox_timer = time.perf_counter()
            I = torch.zeros((self.input_bounds.shape[0], 
                    self.input_bounds.shape[0]+ 1), dtype = Setting.TORCH_PRECISION)
            I = I.fill_diagonal_(1).unsqueeze(0)
            input_bounds = self.input_bounds.unsqueeze(0).unsqueeze(1)
            layer_sym = SymbolicInterval(input_bounds.to(Setting.DEVICE),I,I, device = Setting.DEVICE)
            layer_sym.concretize()
            output_sym = self.int_net(layer_sym)
            output_bounds = output_sym.concrete_bounds.squeeze().to('cpu')
            status, _ = self._check_violated_bounds(self.spec.objectives, output_bounds)
            logger.debug(f"Try overapproximation Verification time: {time.perf_counter()-overapprox_timer:.2f} seconds")
            if(status == ResultType.NO_SOL):
                return status
        
        #Init workers
        num_workers = 1 if Setting.N_VERIF_CORES <= 1 else Setting.N_VERIF_CORES
        num_workers = 1
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
            private_state = WorkerData(i, self.model, self.int_net)
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




