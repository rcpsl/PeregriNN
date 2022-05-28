import torch.nn as nn
from intervals.symbolic_interval import SymbolicInterval
import torch
from copy import deepcopy
class ReLU(nn.Module):
    def __init__(self, torch_layer):
        super().__init__()
        self.torch_layer = torch_layer
        self.pre_symbolic = None #dummy for JIT
        self.post_symbolic = None  #dummy for JIT
        self.active = torch.tensor([])
        self.inactive = torch.tensor([])
        self.unstable = torch.tensor([])
    def forward(self, x: SymbolicInterval) ->SymbolicInterval:
        """
        Parameters
        ----------
        x: Symbolic Interval Object
        """
        self.pre_symbolic = x
        post_interval = SymbolicInterval(x.input_interval, x.l.clone(), x.u.clone())

        inactive_relus_idx = torch.nonzero(x.conc_ub <= 0,as_tuple=True)
        post_interval.l[inactive_relus_idx] = 0
        post_interval.u[inactive_relus_idx] = 0
        self.inactive = inactive_relus_idx

        active_relus_idx = torch.nonzero(x.conc_lb > 0,as_tuple=True)
        post_interval.l[active_relus_idx] = x.l[active_relus_idx]
        post_interval.u[active_relus_idx] = x.u[active_relus_idx]
        self.active = active_relus_idx

        unstable_relus_idx = torch.nonzero((x.conc_lb < 0) * (x.conc_ub > 0),as_tuple = True)
        self.unstable = unstable_relus_idx
        if(len(unstable_relus_idx) != 0):

            unstable_pre_conc_lb = x.conc_lb[unstable_relus_idx]
            unstable_pre_conc_ub = x.conc_ub[unstable_relus_idx]
            # unstable_pre_max_lb = x.max_lb[unstable_relus_idx]

            #The ReLU is inactive for most of the input space
            mostly_inactive = torch.nonzero( (torch.abs(unstable_pre_conc_lb) > torch.abs(unstable_pre_conc_ub)) + (x.max_lb[unstable_relus_idx] <=0), as_tuple= True)
            mostly_inactive = (unstable_relus_idx[0][mostly_inactive],unstable_relus_idx[1][mostly_inactive])
            # mostly_inactive = unstable_relus_idx[mostly_inactive]
            post_interval.l[mostly_inactive] = 0

            mostly_active = torch.nonzero(torch.abs(unstable_pre_conc_lb) <= torch.abs(unstable_pre_conc_ub)).squeeze() 
            mostly_active = (unstable_relus_idx[0][mostly_active],unstable_relus_idx[1][mostly_active])
            # mostly_active = unstable_relus_idx[mostly_active]
            a = x.max_lb[mostly_active] /  (x.max_lb[mostly_active] - x.conc_lb[mostly_active])
            a[x.max_lb[mostly_active] < 0] = 0.
            if(len(a.shape) > 0):
                a = a.unsqueeze(1)
            post_interval.l[mostly_active] = a * x.l[mostly_active]


            #Upper bound approximation
            unstable_pre_min_ub = x.min_ub[unstable_relus_idx]
            zero_crossing = torch.nonzero(unstable_pre_min_ub <= 0).squeeze()
            zero_crossing = (unstable_relus_idx[0][zero_crossing],unstable_relus_idx[1][zero_crossing])
            # zero_crossing = unstable_relus_idx[zero_crossing]
            a = x.conc_ub[zero_crossing] / (x.conc_ub[zero_crossing] - x.min_ub[zero_crossing])
            if(len(a.shape) > 0):
                a = a.unsqueeze(1)
            post_interval.u[zero_crossing] = a *  x.u[zero_crossing]
            post_interval.u[...,-1][zero_crossing] -= a.squeeze() * x.min_ub[zero_crossing]


        post_interval.concretize()
        self.post_symbolic = post_interval
        return post_interval

