import torch
import numpy as np
import logging

class Setting:
    
    #Global
    OMP_NUM_THREADS = 1
    TORCH_PRECISION = torch.float32
    NP_PRECISION = np.float32
    DEVICE = torch.device('cpu')

    #Verifier
    N_VERIF_CORES = 1
    TIMEOUT = 300
    MAX_DEPTH = 300
    TRY_SAMPLING = True
    TRY_OVERAPPROX = True
    N_SAMPLES = 15000

    #Symbolic analysis
    USE_GPU = False

    #Logger
    LOG_LEVEL =  logging.DEBUG