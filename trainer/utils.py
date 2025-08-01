# Implements helper functions.

import random

import numpy as np
import torch

from .configuration import SystemConfig, TrainerConfig, DataloaderConfig



class AverageMeter:
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count



def patch_configs(epoch_num_to_set=TrainerConfig.epoch_num, 
                  batch_size_to_set=DataloaderConfig.batch_size):
    """ 
    
    Patches configs if cuda is not available

    Returns:
        returns patched dataloader_config and trainer_config

    """
    # default experiment params
    num_workers_to_set = DataloaderConfig.num_workers

    if torch.cuda.is_available():
        print("Using CUDA backend for PyTorch")
        device = "cuda"
    #if torch.backends.mps.is_available():
    #    print("Using MPS backend for PyTorch")
    #    device = "mps"
    else:
        raise RuntimeError("No GPU - check your setup.")
        device = "cpu"
        batch_size_to_set = 16
        num_workers_to_set = 2
        epoch_num_to_set = 1

    # Change the settings of dataloader and trainer configs
    dataloader_config = DataloaderConfig(batch_size=batch_size_to_set, 
                                         num_workers=num_workers_to_set)

    trainer_config = TrainerConfig(device=device, 
                                  epoch_num=epoch_num_to_set)
    
    return dataloader_config, trainer_config



def setup_system(system_config: SystemConfig) -> None:
    """
    Generic system configuration: sets random seeds, torch print options, and cudnn settings.
    """
    
    # configure random numbers
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    
    # configure printing options for torch
    torch.set_printoptions(precision=10)
    
    # CuDNN
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic
