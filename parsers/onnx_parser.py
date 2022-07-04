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

    def __init__(self, onnx_path: str, simplify = False) -> None:

        try:
            self.onnx_model = onnx.load(onnx_path)   
            logger.info("Loaded ONNX model")
            if(simplify):
                import dnnv.nn
                from dnnv.nn.transformers.simplifiers import (simplify, ReluifyMaxPool)
                from pathlib import Path
                t = time.perf_counter()
                op_graph = dnnv.nn.parse(Path(onnx_path))
                simplified_model = simplify(op_graph, ReluifyMaxPool(op_graph))
                diff = time.perf_counter() - t
                logger.debug(f"Simplifying ONNX model took: {diff} sec")
                t = time.perf_counter()
                self.onnx_model = simplified_model.as_onnx()
                diff = time.perf_counter() - t
                logger.debug(f"Exporting back to ONNX model took: {diff} sec")

        except ImportError as e:
            logger.exception(str(e))
            logger.info("DNNV not installed. Required for convnets with pooling, BatchNorm layers")
        except Exception as e:
            logger.exception(str(e))
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
        
