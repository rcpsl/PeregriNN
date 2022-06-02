import torch
import numpy as np
class Setting:
    
    #Global
    OMP_NUM_THREADS = 1
    TORCH_PRECISION = torch.float32
    NP_PRECISION = np.float32

    #Verifier
    N_VERIF_CORES = 1
    TIMEOUT = 300
    MAX_DEPTH = 300
    #Symbolic analysis
    USE_GPU = False
