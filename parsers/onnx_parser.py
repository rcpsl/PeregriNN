import onnx2pytorch
import onnx
from utils.Logger import get_logger
logger = get_logger(__name__)

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
            pytorch_model = onnx2pytorch.ConvertModel(self.onnx_model)
            logger.info("Converted ONNX model to Pytorch model")
            
        except Exception as e:
            print(e) 
            logger.exception("Failed to convert ONNX model")
            raise e

        return pytorch_model
        
