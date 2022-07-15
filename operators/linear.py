import torch.nn as nn
import torch.nn.functional as F
from intervals.symbolic_interval import SymbolicInterval
import torch
from copy import deepcopy


class Linear(nn.Module):
    def __init__(self, torch_layer, input_shape = None):
        super().__init__()
        self.torch_layer = torch_layer
        self.pre_symbolic = None
        self.post_symbolic = None
        self.input_shape = input_shape
        self.output_shape = torch.tensor([self.torch_layer.weight.shape[0]])

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

        if b is not None:
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




        

class Conv2d(nn.Module):
    def __init__(self, torch_layer, input_shape = None):
        super().__init__()
        self.torch_layer = torch_layer
        self.pre_symbolic = None
        self.post_symbolic = None
        self.output_shape = None
        self.input_shape = input_shape
        if input_shape is not None:
            self.output_shape = self._get_output_shape(input_shape)
        
    def _get_output_shape(self, input_shape):
        #Assume dim 0 is always the batch size
        Ci, Hi, Wi = input_shape
        Co = self.torch_layer.out_channels
        padding = self.torch_layer.padding
        dilation = self.torch_layer.dilation
        kernel_size = self.torch_layer.kernel_size
        stride = self.torch_layer.stride
        Ho = (Hi + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]
        Wo = (Wi + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]
        Ho = int(Ho + 1)
        Wo = int(Wo + 1)

        return torch.tensor([Co, Ho, Wo])

    def _conv_args(self):
        params = {}
        params['padding'] = self.torch_layer.padding
        params['dilation'] = self.torch_layer.dilation
        # params['weight'] = self.torch_layer.weight
        params['stride'] = self.torch_layer.stride
        params['groups'] = self.torch_layer.groups
        return params

    def forward(self, x : SymbolicInterval) -> SymbolicInterval:
        """
        Parameters
        ----------
        x: Symbolic Interval Object
        """
        self.pre_symbolic = x
        out_shape = self.output_shape
        batch_size = x.shape[0]
        out_flat_shape = torch.prod(out_shape)
        post_interval = SymbolicInterval(x.input_interval, n=out_flat_shape)

        #Stack the symbolic equations in the batch dim 
        c,h,w = self.input_shape
        l_t = x.l.T.reshape(-1, c,h,w)
        u_t = x.u.T.reshape(-1,c,h,w)
        params = self._conv_args()
        W = self.torch_layer.weight
        b = self.torch_layer.bias
        zeros_W = torch.zeros_like(W)
        W_pos = torch.maximum(zeros_W,W)
        W_neg = torch.minimum(zeros_W,W)
        post_l = F.conv2d(l_t, W_pos, **params) + F.conv2d(u_t, W_neg, **params)
        post_u = F.conv2d(l_t, W_neg, **params) + F.conv2d(u_t, W_pos, **params)
        if b is not None:
            post_l[-1,...] += b.reshape((-1,1,1))
            post_u[-1,...] += b.reshape((-1,1,1))
        post_interval.l = post_l.reshape(post_l.shape[0],-1,batch_size).T
        post_interval.u = post_u.reshape(post_u.shape[0],-1,batch_size).T


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

    