'''
Class that converts torch model to interval network. 
Propagating an interval through this network computes bounds on all neurons using an underlying Interval bound propagation method (SIA,IA,..) 
'''
import torch.nn as nn
import torch
from intervals.symbolic_interval import SymbolicInterval
from operators.activations import ReLU
from utils.Logger import get_logger
logger = get_logger(__name__)

class IntervalNetwork(nn.Module):
    def __init__(self, model: nn.Module, input_interval: torch.tensor, operators_dict: dict, in_shape = None):
        super().__init__()
        # self._model = model
        self.operators_dict = operators_dict
        self.input_interval = input_interval
        self.layers = []

        out_shape = in_shape
        for module in list(model.modules())[1:]:
            module_name = str(module).split('(')[0]
            try:
                self.layers.append(self.operators_dict[module_name](module, input_shape = out_shape))
                out_shape = self.layers[-1].output_shape 
            except:
                logger.exception(f"Operation {module_name} not implemented")

       
        self.interval_net = nn.Sequential(*self.layers)

    
    def forward(self, interval : SymbolicInterval, layers_mask : dict = None) -> SymbolicInterval:
        if layers_mask is None:
            return self.interval_net(interval)
        else:
            prev = interval
            for idx, layer in enumerate(self.layers):
                layer_mask = layers_mask[idx]
                if(type(layer) == ReLU):
                    prev = layer(prev, layer_mask)
                else:
                    prev = layer(prev)
            return prev
    
    def evaluate(self, input : torch.tensor) -> torch.tensor:
        prev = input.reshape((-1,*self.layers[0].input_shape))
        for layer in self.layers:
            prev = layer.torch_layer(prev)
        
        return prev

