import torch.nn as nn
from intervals.symbolic_interval import SymbolicInterval
import torch
from copy import deepcopy



class Linear(nn.Module):
    def __init__(self, torch_layer):
        super().__init__()
        self.torch_layer = torch_layer
        self.pre_symbolic = None
        self.post_symbolic = None
        

    def forward(self, x : SymbolicInterval) -> SymbolicInterval:
        """
        Parameters
        ----------
        x: Symbolic Interval Object
        """
        self.pre_symbolic = x
        post_interval = SymbolicInterval(x.input_interval, n=self.torch_layer.out_features)

        W = self.torch_layer.weight
        b = self.torch_layer.bias
        zeros_W = torch.zeros_like(W)
        W_pos = torch.maximum(zeros_W,W)
        W_neg = torch.minimum(zeros_W,W)

        post_interval.l  = W_pos @ x.l + W_neg @ x.u
        post_interval.u  = W_pos @ x.u + W_neg @ x.l

        post_interval.l[...,-1] += b
        post_interval.u[...,-1] += b

        post_interval.concretize()
        self.post_symbolic = post_interval
        return post_interval

    @property
    def pre_conc_bounds(self):
        return self.pre_symbolic.conc_bounds

    @property 
    def post_conc_bounds(self):
        return self.post_symbolic.conc_bounds




        

