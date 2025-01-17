import os
import sys 
import logging

"""
Logger to log steps wherever it is required.
"""

logging_str= "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir= "logs"

log_filepath = os.path.join(log_dir, "running_log.log")
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    level= logging.INFO,
    format= logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Rag_Logger")