from typing import Callable, Iterable
from dataclasses import dataclass

from torchvision.transforms import ToTensor



@dataclass
class SystemConfig:
    seed: int = 10  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)



@dataclass
class DatasetConfig:
    root_dir: str = "data"  # dataset directory root
    train_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during training data preparation
    test_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during test data preparation



@dataclass
class DataloaderConfig:
    batch_size: int =  64 #250  # amount of data to pass through the network at each forward-backward iteration
    num_workers: int = 12 #5  # number of concurrent processes using to prepare data
    test_size: float = 0.2  # proportion of the dataset to use for validation
    persistent_workers: bool = False  # whether to keep dataloader workers alive after each epoch



@dataclass
class OptimizerConfig:
    
    learning_rate: float = 0.0005  # determines the speed of network's weights update
    momentum: float = 0.8  # used to improve vanilla SGD algorithm and provide better handling of local minimas
    weight_decay: float = 4e-3 #1e-3  # amount of additional regularization on the weights values
    
    lr_step_milestones: Iterable = (
        30, 40
    )  # at which epoches should we make a "step" in learning rate (i.e. decrease it in some manner)
    
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.8  # multiplier applied to current learning rate at each of lr_ctep_milestones



@dataclass
class TrainerConfig:
    model_dir: str = "checkpoints"  # directory to save model states
    model_saving_frequency: int = 1  # frequency of model state savings per epochs
    device: str = "cuda"  # device to use for training.
    epoch_num: int = 80 # 200 #80  # number of times the whole dataset will be passed through the network
    progress_bar: bool = True  # enable progress bar visualization during train process
