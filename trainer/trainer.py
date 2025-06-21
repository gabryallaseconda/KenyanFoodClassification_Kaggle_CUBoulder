"""
Trainin pipeline for the model.
"""

import os
import datetime

from typing import Union, Callable
from pathlib import Path
from operator import itemgetter

import torch

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .hooks import test_hook_default, train_hook_default
from .visualizer import Visualizer



class Trainer:  # pylint: disable=too-many-instance-attributes
    """ Generic class for training loop.

    Parameters
    ----------
    model : nn.Module
        torch model to train
    loader_train : torch.utils.DataLoader
        train dataset loader.
    loader_test : torch.utils.DataLoader
        test dataset loader
    loss_fn : callable
        loss function
    metric_fn : callable
        evaluation metric function
    optimizer : torch.optim.Optimizer
        Optimizer
    lr_scheduler : torch.optim.LrScheduler
        Learning Rate scheduler
    configuration : TrainerConfiguration
        a set of training process parameters
    data_getter : Callable
        function object to extract input data from the sample prepared by dataloader.
    target_getter : Callable
        function object to extract target data from the sample prepared by dataloader.
    visualizer : Visualizer, optional
        shows metrics values (various backends are possible)
    # """
    def __init__( # pylint: disable=too-many-arguments
        self,
        model: torch.nn.Module,
        loader_train: torch.utils.data.DataLoader,
        loader_test: torch.utils.data.DataLoader,
        loss_fn: Callable,
        metric_fn: Callable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Callable,
        device: Union[torch.device, str] = "cuda",
        model_saving_frequency: int = 1,
        save_dir: Union[str, Path] = "checkpoints",
        model_name_prefix: str = "model",
        data_getter: Callable = itemgetter("image"),
        target_getter: Callable = itemgetter("target"),
        stage_progress: bool = True,
        visualizer: Union[Visualizer, None] = None,
        get_key_metric: Callable = itemgetter("top1"),
    ):
        
        self.model = model
        
        self.loader_train = loader_train
        self.loader_test = loader_test
        
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.device = device
        
        self.model_saving_frequency = model_saving_frequency
        self.save_dir = save_dir
        self.model_name_prefix = model_name_prefix
        
        self.data_getter = data_getter
        self.target_getter = target_getter
        
        self.stage_progress = stage_progress
        
        self.visualizer = visualizer
        self.get_key_metric = get_key_metric
        
        # Intrnal attributes - tracking of metrics
        self.metrics = {"epoch": [], "train_loss": [], "test_loss": [], "test_metric": []}
        
        # Internal attributes - hooks for training and testing
        self.hooks = {} 
        self._register_default_hooks()
        
    
    def _register_default_hooks(self):
        """
        Register initial hooks for training and testing.
        """
        
        self.hooks["train"] = train_hook_default
        self.hooks["test"] = test_hook_default
        self.hooks["end_epoch"] = None
        

    def fit(self, epochs):
        """ 
        Fit model method.

        Arguments:
            epochs (int): number of epochs to train model.
        """
        
        # Tqdm progress bar for epochs
        iterator = tqdm(range(epochs), dynamic_ncols=True)
        
        for epoch in iterator:
            
            # Run the train procedure (all in the hoook)
            output_train = self.hooks["train"](
                self.model,
                self.loader_train,
                self.loss_fn,
                self.optimizer,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter
            )
            
            # Run the test procedure (all in the hoook)
            output_test = self.hooks["test"](
                self.model,
                self.loader_test,
                self.loss_fn,
                self.metric_fn,
                self.device,
                prefix="[{}/{}]".format(epoch, epochs),
                stage_progress=self.stage_progress,
                data_getter=self.data_getter,
                target_getter=self.target_getter,
                get_key_metric=self.get_key_metric
            )
            
            # Update the visualizer with the current metrics (tensorboard)
            if self.visualizer:
                self.visualizer.update_charts(
                    train_metric = None,
                    train_loss = output_train['loss'], 
                    test_metric = output_test['metric'], 
                    test_loss = output_test['loss'],
                    learning_rate = self.optimizer.param_groups[0]['lr'], 
                    epoch = epoch
                )

            # Update the metrics
            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(output_train['loss'])
            self.metrics['test_loss'].append(output_test['loss'])
            self.metrics['test_metric'].append(output_test['metric'])

            # Update the scheduler if it is defined
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(output_train['loss'])
                else:
                    self.lr_scheduler.step()

            # Execute the end of epoch hook if it is defined
            if self.hooks["end_epoch"] is not None:
                self.hooks["end_epoch"](iterator, epoch, output_train, output_test)

            # Save the model state if the epoch is a multiple of the saving frequency
            if (epoch + 1) % self.model_saving_frequency == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, self.model_name_prefix) + str(datetime.datetime.now())
                )
                
            # Update the progress bar description
            iterator.set_description(
                "[{}/{}] Train Loss: {:.5f}, Test Loss: {:.5f}, Test Metric: {}".format(
                    epoch + 1, epochs, output_train['loss'], output_test['loss'], output_test['metric']
                )
            )
            
        return self.metrics




