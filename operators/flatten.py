import torch.nn as nn
from intervals.symbolic_interval import SymbolicInterval
import torch
from copy import deepcopy



class Flatten(nn.Module):
    def __init__(self, torch_layer, input_shape = None):
        super().__init__()
        self.torch_layer = torch_layer
        self.pre_symbolic = None
        self.post_symbolic = None
        self.input_shape = input_shape
        self.output_shape = None
        if input_shape is not None:
            self.output_shape = torch.prod(input_shape)
        
        

    def forward(self, x : SymbolicInterval):
        """
        Parameters
        ----------
        x: Symbolic Interval Object
        """
        self.pre_symbolic = x
        post_interval = SymbolicInterval(x.input_interval, x.l.clone(), x.u.clone())
        post_interval.concretize()
        self.post_symbolic = post_interval
        return post_interval

    @property
    def pre_conc_bounds(self):
        return self.pre_symbolic.conc_bounds
    
    @property 
    def post_conc_bounds(self):
        return self.post_symbolic.conc_bounds

    @property 
    def post_conc_lb(self):
        return self.post_symbolic.conc_lb

    @property 
    def post_conc_ub(self):
        return self.post_symbolic.conc_ub
    
    @property 
    def pre_conc_lb(self):
        return self.pre_symbolic.conc_lb

    @property 
    def pre_conc_ub(self):
        return self.pre_symbolic.conc_ub


class Reshape(nn.Module):
    def __init__(self, torch_layer, input_shape = None):
        super().__init__()
        self.torch_layer = torch_layer
        self.pre_symbolic = None
        self.post_symbolic = None
        self.input_shape = input_shape
        self.output_shape = self.torch_layer.shape[1:]
  
        
        

    def forward(self, x : SymbolicInterval):
        """
        Parameters
        ----------
        x: Symbolic Interval Object
        """
        self.pre_symbolic = x
        post_interval = SymbolicInterval(x.input_interval, x.l.clone(), x.u.clone())
        post_interval.concretize()
        self.post_symbolic = post_interval
        return post_interval

    @property
    def pre_conc_bounds(self):
        return self.pre_symbolic.conc_bounds
    
    @property 
    def post_conc_bounds(self):
        return self.post_symbolic.conc_bounds

    @property 
    def post_conc_lb(self):
        return self.post_symbolic.conc_lb

    @property 
    def post_conc_ub(self):
        return self.post_symbolic.conc_ub
    
    @property 
    def pre_conc_lb(self):
        return self.pre_symbolic.conc_lb

    @property 
    def pre_conc_ub(self):
        return self.pre_symbolic.conc_ub