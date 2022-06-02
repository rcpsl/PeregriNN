
import torch
class Dataset_MetaData:
    inout_shapes = {
        'cifar20': {'input':torch.tensor([3,32,32], dtype = torch.int), 
                    'output':torch.tensor([10], dtype = torch.int)},
        'mnistfc': {'input':torch.tensor([784], dtype = torch.int), 
                    'output':torch.tensor([10], dtype = torch.int)}
        }