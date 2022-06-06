'''
Haitham Khedr

File containing the main code for the NN verifier 
'''
from ctypes import c_bool
import multiprocessing as mp 
import torch.nn as nn
import torch
from intervals.symbolic_interval import SymbolicInterval

from utils.config import Setting
from intervals.interval_network import IntervalNetwork
from operators import registered_ops as OPS
from utils.sample_network import sample_network
from utils.specification import Specification

import time
from enum import Enum

from utils.Logger import get_logger
logger = get_logger(__name__)

class VerificationResult(Enum):
    SOL_FOUND = 0
    NO_SOL = 1
    TIMEOUT = 2
    UNKNOWN = 3

class Worker(mp.Process):
    def __init__(self, name, shared, private):
        self.name       = name
        self._shared    = shared
        self._private   = private

    def run(self):
        pass

class Branch:
    '''
    Defines a computational branch used in verification
    '''
    def __init__(self, spec: Specification) -> None:
        self.input_bounds = spec.input_bounds
        self.spec = spec
        self.fixed_neurons = [] #list of tuples (layer_idx, neuron_idx, phase)
        self.verification_result = VerificationResult.UNKNOWN

    def fix_neuron(self, layer_idx, neuron_idx, phase):
        self.fixed_neurons.append((layer_idx, neuron_idx, phase))

    def clone(self):
        new_branch = Branch(self.input_bounds, self.spec)
        new_branch.fixed_neurons = self.fixed_neurons.copy()
        return new_branch   

class SharedData():
    def __init__(self, model, interval_net, spec):
        self.model      = model
        self.int_net    = interval_net
        self.spec       = spec

        self.task_Q = mp.Queue()
        self.poison_pill = mp.Value(c_bool, False)
        self.n_branches = mp.Value('I', 0)
        

class WorkerData():
    def __init__(self):
        pass

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
        self.verification_result = VerificationResult.UNKNOWN

    def _check_violated_bounds(self, objectives : list, bounds : torch.tensor) -> tuple[VerificationResult,torch.tensor]:
        for A,b in objectives:
            A = torch.from_numpy(A).type(Setting.TORCH_PRECISION)
            b = torch.from_numpy(b).type(Setting.TORCH_PRECISION)
            zeros = torch.zeros_like(A)
            A_pos = torch.maximum(zeros, A)
            A_neg = torch.minimum(zeros, A)
            lower_bound = A_pos @ bounds[:,0] + A_neg @ bounds[:,1]
            if(lower_bound > b):
                return (VerificationResult.NO_SOL, torch.tensor([]))
            else:
                return (VerificationResult.UNKNOWN, torch.tensor([]))


    def _check_violated_samples(self, objectives : list, outputs : torch.tensor) -> tuple[VerificationResult,torch.tensor]:
        for A,b in objectives:
            counterexamples = torch.where(outputs @ A.T < torch.from_numpy(b))[0]
            if(len(counterexamples) > 0):
                return (VerificationResult.SOL_FOUND, outputs[0])

        return (VerificationResult.UNKNOWN, torch.tensor([]))

    def verify(self, timeout : float) -> VerificationResult:
        #TODO: Set SIGALARM handler
        start = time.perf_counter()
        if Setting.TRY_SAMPLING:
            sampling_timer = time.perf_counter() 
            in_bounds_reshaped = self.input_bounds.reshape(*self.spec.input_shape ,2)
            samples = sample_network(in_bounds_reshaped, Setting.N_SAMPLES).to(Setting.DEVICE)
            outputs = self.model(samples).to('cpu')
            status,ce = self._check_violated_samples(self.spec.objectives, outputs)
            logger.debug(f"Sampling verification time: {time.perf_counter()-sampling_timer:.2f} seconds")
            if(status == VerificationResult.SOL_FOUND):
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
            if(status == VerificationResult.NO_SOL):
                return status
        
        #Init workers
        num_workers = 1 if Setting.N_VERIF_CORES <= 1 else Setting.N_VERIF_CORES


        end = time.perf_counter()
        logger.debug(f"Total Verification time: {end-start:.2f} seconds")




