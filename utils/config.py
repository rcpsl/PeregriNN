import os
import torch
import numpy as np
import logging

from utils.datasets_info import Dataset_MetaData

class Setting:
    
    #Global
    OMP_NUM_THREADS = 1
    TORCH_PRECISION = torch.float64
    NP_PRECISION = np.float64
    DEVICE = torch.device('cpu')
    EPS = 1E-10

    #Verifier
    N_VERIF_CORES = 32
    TIMEOUT = 300
    MAX_DEPTH = 300
    TRY_SAMPLING = True
    TRY_OVERAPPROX = True
    N_SAMPLES = 15000
    ONLY_FIRST_INFEASIBLE_LAYER = True

    #Symbolic analysis
    USE_GPU = False

    #Logger
    LOG_LEVEL =  logging.INFO


def update_cfg(args):
    category = args.category
    args.dataset = category
    if(category == 'rl_benchmarks'):
        Setting.EPS = 0.0
        Setting.N_VERIF_CORES = 24
    if(category == 'mnistfc'):
        args.subtract_target = True
    if(category == 'oval21'):
        args.subtract_target = True
        args.dataset = 'cifar20'
        Setting.N_VERIF_CORES = 24
    if(category == 'cifar_biasfield'):
        model_name = args.model.split('/')[-1].split('.')[0]
        model_name_simplified = model_name + '_simplified.onnx'
        simplified_path = os.path.dirname(os.path.dirname(__file__)) 
        simplified_path = os.path.join(simplified_path,'vnn_simplified_networks',model_name_simplified)
        args.model = simplified_path
        Setting.N_VERIF_CORES = 24
        Setting.TRY_SAMPLING = False
    if(category == 'collins_rul_cnn'):
        Setting.EPS  = 0.0
        fname = args.model
        if 'full_window_40' in fname:
            Dataset_MetaData.inout_shapes[args.dataset]['input'] = torch.tensor([1,40,20], dtype = torch.int)
    if(category == 'reach_prob_density'):
        Setting.EPS  = 0.0
        Setting.N_VERIF_CORES = 24
        model_name = args.model
        if('vdp' in model_name):
            args.dataset = 'reach_prob_density_vdp'
        elif('robot' in model_name):
            args.dataset = 'reach_prob_density_robot'
        else:
            args.dataset = 'reach_prob_density_gcas'
    
    if(category == 'rl_benchmarks'):
        model_name = args.model
        if('dubins' in model_name):
            args.dataset = 'rl_benchmarks_dubinsrejoin'
        elif('cartpole' in model_name):
            args.dataset = 'rl_benchmarks_cartpole'
        else:
            args.dataset = 'rl_benchmarks_lunarlander'

    if(category =='tllverifybench'):
        Setting.N_VERIF_CORES = 24

    if 'test/test_' in args.model:
        args.dataset = 'test'
        