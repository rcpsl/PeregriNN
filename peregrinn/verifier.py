'''
Haitham Khedr

File containing the main code for the NN verifier 
'''
import torch.nn as nn
import torch
from peregrinn.branch import Branch

from utils.config import Setting
from intervals.interval_network import IntervalNetwork
from operators import registered_ops as OPS
from utils.specification import Specification

from enum import Enum

class VerificationResult(Enum):
    SOL_FOUND = 0
    NO_SOL = 1
    TIMEOUT = 2
    UNKNOWN = 3

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
        self.spec = vnnlib_spec
        self.int_net = IntervalNetwork(self.model, self.input_bounds, 
                            operators_dict=OPS.op_dict, in_shape = self.spec.input_size)
        
        self.init_branch = Branch()
        self.verification_result = VerificationResult.UNKNOWN
        pass
