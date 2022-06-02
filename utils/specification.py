import torch
from utils.Logger import get_logger
from utils.config import Setting
from utils.datasets_info import Dataset_MetaData

logger = get_logger(__name__)

class Specification:
    def __init__(self, spec, dataset='mnistfc'):
        self.spec = spec
        self.dataset = dataset
        try:
            self.input_bounds = torch.tensor(self.spec[0][0],dtype = Setting.TORCH_PRECISION)
            self.objectives = self.spec[0][1]
        except Exception as e:
            logger.exception("Failed to read VNNLIB specification. Error message:" + e)
            raise
            
        try:
            self.input_size = Dataset_MetaData.input_size[dataset]
        except Exception as e:
            logger.exception(e)
            raise