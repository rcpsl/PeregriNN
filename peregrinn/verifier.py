'''
Haitham Khedr

File containing the main code for the NN verifier 
'''
import torch.nn as nn
import torch

from utils.config import Setting
from intervals.interval_network import IntervalNetwork
from operators import registered_ops as OPS
from utils.sample_network import sample_network
from utils.specification import Specification

from enum import Enum

class VerificationResult(Enum):
    SOL_FOUND = 0
    NO_SOL = 1
    TIMEOUT = 2
    UNKNOWN = 3

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
                            operators_dict=OPS.op_dict, in_shape = self.spec.input_shape)
        
        self.init_branch = Branch(self.spec)
        self.verification_result = VerificationResult.UNKNOWN

    def _check_violated(self, objectives, outputs):
        for A,b in objectives:
            counterexamples = torch.where(outputs @ A.T < torch.from_numpy(b))[0]
            if(len(counterexamples) > 0):
                return True, outputs[0]

        return False
    def verify(self) -> VerificationResult:

        if Setting.TRY_SAMPLING:
            #TODO: Sample N_SAMPLES and verify property
            in_bounds_reshaped = self.input_bounds.reshape(*self.spec.input_shape ,2)
            samples = sample_network(in_bounds_reshaped, Setting.N_SAMPLES)
            outputs = self.model(samples)
            violated = self._check_violated(self.spec.objectives, outputs)
            if(violated):
                return VerificationResult.SOL_FOUND
            pass


