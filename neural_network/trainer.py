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

from neural_network.hooks import test_hook_default, train_hook_default

from neural_network.configuration import systemConfig, trainingConfig

class Trainer:  
    def __init__(
        self,
        model: torch.nn.Module,
        
        loader_train: torch.utils.data.DataLoader,
        loader_test: torch.utils.data.DataLoader,
        
        loss_fn: Callable,
        metric_fn: Callable,
        
        optimizer: torch.optim.Optimizer,
        scheduler: Callable,
        
        visualizer = None,
    ):
        
        self.model = model
        
        self.loader_train = loader_train
        self.loader_test = loader_test
        
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.visualizer = visualizer
        

        self.device = systemConfig.device
        self.epochs = trainingConfig.number_of_epochs
        
        self.model_saving_frequency = trainingConfig.model_saving_frequency
        self.save_dir = trainingConfig.model_dir
        self.model_name_prefix = trainingConfig.model_name_prefix
        
        self.data_getter = lambda sample: sample["image"]
        self.target_getter = lambda sample: torch.tensor(sample["target"])
        
        self.stage_progress = trainingConfig.progress_bar_on_batches_inside_epoch

        self.get_key_metric = lambda metric: metric["top1"]
        
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
        

    def fit(self):

        iterator = tqdm(range(self.epochs), dynamic_ncols=True)
        
        for epoch in iterator:
            
            # Run the train procedure (all in the hoook)
            output_train = self.hooks["train"](
                self.model,
                self.loader_train,
                self.loss_fn,
                self.optimizer,
                self.device,
                prefix="[{}/{}]".format(epoch, self.epochs),
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
                prefix="[{}/{}]".format(epoch, self.epochs),
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



            self._do_scheduler_step(output_train)

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
                    epoch + 1, 
                    self.epochs, 
                    output_train['loss'], 
                    output_test['loss'], 
                    output_test['metric']
                )
            )
            
        return self.metrics



    def _do_scheduler_step(self, output_train):
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(output_train['loss'])
            else:
                self.scheduler.step()
