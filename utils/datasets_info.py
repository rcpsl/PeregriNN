
import torch
class Dataset_MetaData:
    inout_shapes = {
        'cifar20': {'input':torch.tensor([3,32,32], dtype = torch.int), 
                    'output':torch.tensor([10], dtype = torch.int)},
        'mnistfc': {'input':torch.tensor([784], dtype = torch.int), 
                    'output':torch.tensor([10], dtype = torch.int)},
        'imagenet': {'input':torch.tensor([3,224,224], dtype = torch.int), 
                    'output':torch.tensor([1000], dtype = torch.int)},
        'collins': {'input':torch.tensor([1,20,20], dtype = torch.int), 
                    'output':torch.tensor([1], dtype = torch.int)},
        'reach_prob': {'input':torch.tensor([3], dtype = torch.int), 
                    'output':torch.tensor([3], dtype = torch.int)}
        }