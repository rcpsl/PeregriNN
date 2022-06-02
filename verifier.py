'''
Haitham Khedr

File containing the main code for the NN verifier 
'''
import torch.nn as nn
import torch

from utils.config import Setting
from intervals.interval_network import IntervalNetwork
from operators import registered_ops as OPS
class Verifier:
    '''
    A class that is used for the Verification of a given spec

    Attributes
    ----------

    Methods
    -------
    '''

    def __init__(self, model: nn.Module, vnnlib_spec: list):

        self.model = model
        self.spec = vnnlib_spec
        self.input_bounds = torch.tensor(vnnlib_spec[0][0],dtype = Setting.TORCH_PRECISION)
        self.int_net = IntervalNetwork(self.model, self.input_bounds, operators_dict=OPS.op_dict, in_shape = (3,32,32))
        pass
