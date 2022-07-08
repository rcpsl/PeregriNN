import torch
import numpy as np
import logging

class Setting:
    
    #Global
    OMP_NUM_THREADS = 1
    TORCH_PRECISION = torch.float64
    NP_PRECISION = np.float32
    DEVICE = torch.device('cpu')
    EPS = 1E-10

    #Verifier
    N_VERIF_CORES = 16
    TIMEOUT = 300
    MAX_DEPTH = 300
    TRY_SAMPLING = True
    TRY_OVERAPPROX = True
    N_SAMPLES = 15000
    ONLY_FIRST_INFEASIBLE_LAYER = True

    #Symbolic analysis
    USE_GPU = False

    #Logger
    LOG_LEVEL =  logging.DEBUG