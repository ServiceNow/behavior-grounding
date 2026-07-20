import json
import os
import hydra
from omegaconf import DictConfig
import logging
from datetime import datetime


def setup_logging(cfg: DictConfig):
    """Setup logging to both console and file"""
    if not os.path.exists(cfg.experiment.output_dir):
        os.makedirs(cfg.experiment.output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(cfg.experiment.output_dir, f"debug_log_{timestamp}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ],
        force=True  # Force reconfiguration
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    
    return logger
