import logging
from stat import filemode
from statistics import mode
import sys
import os
from utils.config import Setting

    
def get_logger(name: str, f_path :str = "" , log_level = Setting.LOG_LEVEL, propagate = True, handlers = []) -> logging.Logger:
    """
    Constructs a new logger 
    
    Parameters
    ----------
    name: name of the logger
    f_path: path to the output file
    log_level: logging level

    Returns
    -------
    logger: A logger module
    """

    logger = logging.getLogger(name)
    if(log_level):
        logger.setLevel(log_level)
    for handler in handlers:
        logger.addHandler(handler)
    if(f_path):
        log_fh = logging.FileHandler(f_path, encoding='utf-8')
        logger.addHandler(log_fh)

    logger.propagate = propagate

    return logger