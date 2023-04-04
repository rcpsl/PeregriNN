
import torch
class Dataset_MetaData:
    inout_shapes = {
        'cifar20': {'input':torch.tensor([3,32,32], dtype = torch.int), 
                    'output':torch.tensor([10], dtype = torch.int)},
        'mnist_fc': {'input':torch.tensor([784], dtype = torch.int), 
                    'output':torch.tensor([10], dtype = torch.int)},
        'imagenet': {'input':torch.tensor([3,224,224], dtype = torch.int), 
                    'output':torch.tensor([1000], dtype = torch.int)},
        'collins_rul_cnn': {'input':torch.tensor([1,20,20], dtype = torch.int), 
                    'output':torch.tensor([1], dtype = torch.int)},
        'reach_prob_density_vdp': {'input':torch.tensor([3], dtype = torch.int), 
                    'output':torch.tensor([3], dtype = torch.int)},
        'reach_prob_density_robot': {'input':torch.tensor([5], dtype = torch.int), 
                    'output':torch.tensor([5], dtype = torch.int)},
        'reach_prob_density_gcas': {'input':torch.tensor([14], dtype = torch.int), 
                    'output':torch.tensor([14], dtype = torch.int)},
        'cifar_biasfield': {'input':torch.tensor([16], dtype = torch.int), 
                    'output':torch.tensor([10], dtype = torch.int)},
        'rl_benchmarks_cartpole': {'input':torch.tensor([4], dtype = torch.int), 
                    'output':torch.tensor([2], dtype = torch.int)},
        'rl_benchmarks_dubinsrejoin': {'input':torch.tensor([8], dtype = torch.int), 
                    'output':torch.tensor([8], dtype = torch.int)},
        'rl_benchmarks_lunarlander': {'input':torch.tensor([8], dtype = torch.int), 
                    'output':torch.tensor([4], dtype = torch.int)},
        'tllverifybench': {'input':torch.tensor([3], dtype = torch.int), 
                    'output':torch.tensor([1], dtype = torch.int)}
        }