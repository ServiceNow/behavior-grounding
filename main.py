import hydra
from omegaconf import DictConfig
import random
import numpy as np
import torch
import os


def seed_everything(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    seed_everything(cfg.experiment.seed)
    entry_point = hydra.utils.get_method(cfg.task.entry_point)
    entry_point(cfg)



if __name__ == "__main__":
    main()