import onnx2pytorch
import onnx
from utils.Logger import get_logger
import torch.nn as nn
logger = get_logger(__name__)
import time

class ONNX_Parser():
    """
    A class for loading ONNX model and convert it to Pytorch.
    """

    def __init__(self, onnx_path: str) -> None:

        try:
            self.onnx_model = onnx.load(onnx_path)   
            logger.info("Loaded ONNX model")
        except Exception as e:
            print(e) 
            logger.exception(e)
            raise e

    def to_pytorch(self) -> None:

        try:
            s_time = time.perf_counter()
            pytorch_model = onnx2pytorch.ConvertModel(self.onnx_model)
            pytorch_model = nn.Sequential(*list(pytorch_model.modules())[1:])
            logger.info("Converted ONNX model to Pytorch model")
            logger.debug(f"ONNX -> PyTorch conversion time: {time.perf_counter() - s_time:.2f} seconds")
            
        except Exception as e:
            print(e) 
            logger.exception("Failed to convert ONNX model")
            raise e

        return pytorch_model
        
