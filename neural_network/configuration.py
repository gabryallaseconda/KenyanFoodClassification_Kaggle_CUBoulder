import os
from typing import Callable, Iterable
from dataclasses import dataclass

import torch
from torchvision.transforms import ToTensor

import random
import numpy as np



@dataclass
class systemConfig:
    seed: int = 10  
    device: str = ""



def system_setup() -> None:
    torch.set_printoptions(precision=10) # Set precision for tensor printing
    systemConfig.device = _check_device_availability()
    _setup_cudnn()
    _set_seeds()   

    if not trainingConfig.run_test_process:
        modelConfig.run_name = "no_test"

    if trainingConfig.single_batch_overfitting:
        trainingConfig.run_test_process = False
        modelConfig.run_name = "single_batch_overfitting"

    return

def _check_device_availability():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def _setup_cudnn():
    if systemConfig.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

def _set_seeds():
    torch.manual_seed(systemConfig.seed)
    np.random.seed(systemConfig.seed)
    random.seed(systemConfig.seed)
    if systemConfig.device == "cuda":
        torch.cuda.manual_seed_all(systemConfig.seed)



@dataclass
class dataConfig:
    data_root: str = "data"  
    label_file: str = os.path.join(data_root, "train.csv")  
    image_directory: str = os.path.join(data_root, 'images/images') 
    submission_label_file: str = os.path.join(data_root, "test.csv")
    submission_image_directory: str = os.path.join(data_root, 'images/images')

    seed: int = 10  
    
    test_size: float = 0.2
    strategy: bool = True  

    batch_size: int = 64
    
    data_augmentation: bool = True
    resized_image_width: int = 320
    resized_image_height: int = 320

    num_workers: int = 12
    persistent_workers: bool = False   



@dataclass
class modelConfig:
    
    model: str = "modest_cnn" #"large_cnn"
    run_name: str = "training_run"
    
    set_heads_weights_bias_according_to_class_distribution: bool = True  

    model_saving_path = "models_saved/"
    model_prefix = "kenyanfood"

    # TODO: creare un interfaccia export e usare config per decidere quali tipi esportare!)


@dataclass
class trainingConfig:
    run_test_process: bool = True   # TODO: implement this
    single_batch_overfitting: bool = True

    number_of_epochs: int = 300

    #
    optimizer: str = "adam"        # choose from: "adam", "adamw", "sdg"

    learning_rate: float = 3e-4     # adam, adamw, sdg
    momentum: float = 0.8           # sdg
    weight_decay: float = 0 #4e-3      # adam, adamw, sdf

    #
    scheduler: str = "step_lr"  
    
    scheduler_step_size: int = 5  
    scheduler_gamma: float = 0.95  # the higher, the more gentle is the reduction


    model_saving_frequency: int = 4000  # frequency of model state savings per epochs
    model_dir: str = "checkpoints"  # directory to save model states
    model_name_prefix: str = "kenyanfood_model"  # prefix for model state files
    
    
    progress_bar_on_batches_inside_epoch: bool = True


@dataclass
class inferenceConfig:
    score_submission: bool = False  #TODO: qui dovrebbe essere linkato a test/validation nomenclature

    data_root: str = "data"  
    submission_label_file: str = os.path.join(data_root, "test.csv")
    submission_image_directory: str = os.path.join(data_root, 'images/images')
    inference_on_submission_output_path: str = "submission"
    inference_on_submission_output_file: str = "submission.csv"


@dataclass
class metricsConfig:

    is_accuracy_enabled: bool = True
    accuracy_topk: tuple = (1,2,3,5, )

    is_confusion_matrix_enabled: bool = True
    confusion_matrix_normalize: bool = False

    is_precision_recall_enabled: bool = True
    precision_recall_aggregation: str = "macro"
    precision_recall_as_percent: bool = True

    type_scalar: tuple[str, ...] = tuple(
        [f"accuracy_top{k}" for k in (1, 2, 3, 5)] + ["precision", "recall", "f1"]
    )
    type_figure: tuple[str, ...] = ("confusion_matrix_fig",)

