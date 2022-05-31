
import torch


from utils.Logger import get_logger
logger = get_logger(__name__)

class SymbolicInterval():
    def __init__(self, input_interval: torch.Tensor, l :torch.Tensor = [], u : torch.Tensor = [], n :int =0, device : torch.device = 'cpu'):
        self.input_interval = input_interval
        batch_size  = self.input_interval.shape[0]
        if(len(l) > 0 and len(u) > 0):
            self.l = l
            self.u = u
            if(self.l.requires_grad == False):
                self.l = l.requires_grad_()
            if(self.u.requires_grad == False):
                self.u = u.requires_grad_()
        elif(n > 0):
            self.l = torch.zeros(batch_size, n, self.input_interval.shape[2]+1, requires_grad= True)
            self.u = torch.zeros_like(self.l)
        else:
            logger.error("Not enough info to construct a Symbolic interval.")
            raise Exception("Not enough info to construct a Symbolic interval.")

        int_dev = self.input_interval.device
        if(device != int_dev):
            device = int_dev
        self.l = self.l.to(device)
        self.u = self.u.to(device)
        self.input_interval = self.input_interval.to(device)
        self.conc_bounds = torch.zeros(batch_size,2,*self.l.shape[1:-1], 2, device = device)

    
    def zeros_like(self):
        l = torch.zeros_like(self.l)
        u = torch.zeros_like(self.u)
        return SymbolicInterval(self.input_interval,l,u)


    def concretize(self):
        # if(len(self.l) == 0):
        #     logger.exception("Symbolic lower interval is not set. Can't concretize!")

        self._concretize_lower()
        self._concretize_upper()

        return self.conc_bounds.clone()

    def _concretize_lower(self):
        pos_l = torch.maximum(torch.zeros_like(self.l), self.l)
        neg_l = torch.minimum(torch.zeros_like(self.l), self.l)
        self.conc_bounds[:,0,:,0] = torch.sum(pos_l[...,:-1] * self.input_interval[...,0] + neg_l[...,:-1] * self.input_interval[...,1], dim = -1) + self.l[...,-1]
        self.conc_bounds[:,0,:,1] = torch.sum(pos_l[...,:-1] * self.input_interval[...,1] + neg_l[...,:-1] * self.input_interval[...,0], dim = -1) + self.l[...,-1]

    def _concretize_upper(self):

        pos_u = torch.maximum(torch.zeros_like(self.u), self.u)
        neg_u = torch.minimum(torch.zeros_like(self.u), self.u)
        self.conc_bounds[:,1,:,0] = torch.sum(pos_u[...,:-1] * self.input_interval[...,0] + neg_u[...,:-1] * self.input_interval[...,1], dim = -1) + self.u[...,-1]
        self.conc_bounds[:,1,:,1] = torch.sum(pos_u[...,:-1] * self.input_interval[...,1] + neg_u[...,:-1] * self.input_interval[...,0], dim = -1) + self.u[...,-1]
    
        
    @property
    def conc_lb(self):
        return self.conc_bounds[:,0,:,0]
    
    @property
    def conc_ub(self):
        return self.conc_bounds[:,1,:,1]
    
    @property
    def max_lb(self):
        return self.conc_bounds[:,0,:,1]
    
    @property
    def min_ub(self):
        return self.conc_bounds[:,1,:,0]

    @property
    def shape(self):
        return self.l.shape