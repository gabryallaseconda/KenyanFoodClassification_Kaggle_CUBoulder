import os
import datetime
from typing import Union, Callable
from pathlib import Path

from tqdm.auto import tqdm

import torch
from torch.utils.data._utils.collate import default_collate #trasform a list of data into a proper batch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from neural_network.configuration import systemConfig, trainingConfig, dataConfig
from neural_network.logging.tensorboard_interface import TensorBoardInterface
from neural_network.training.metrics import AverageMeter, AccuracyEstimator, metric_epoch_orchestrator


class Trainer:  
    def __init__(self,
                    model: torch.nn.Module,
                    
                    loader_train: torch.utils.data.DataLoader,
                    loader_test: torch.utils.data.DataLoader,
                    
                    loss_function: Callable,
                    
                    optimizer: torch.optim.Optimizer,
                    scheduler: Callable,
                    
                    tensorboard: TensorBoardInterface):
        
        self.model = model
        
        self.loader_train = loader_train
        self.loader_test = loader_test
        
        self.loss_function = loss_function
        
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.tensorboard = tensorboard

        self.epochs = trainingConfig.number_of_epochs
        self.model_saving_frequency = trainingConfig.model_saving_frequency
        self.save_dir = trainingConfig.model_dir
        self.model_name_prefix = trainingConfig.model_name_prefix
        
        self.data_getter = lambda sample: sample["image"]
        self.target_getter = lambda sample: torch.tensor(sample["target"]) #TODO check this as it rise a UserWarning, recommendig to use .clone() for copy construct a tensor
        self.get_key_metric = lambda metric: metric["top1"]
        
        # Intrnal attributes - tracking of metrics
        self.metrics = {"epoch": [], 
                        "metrics_train": [], 
                        "metrics_test": [], 
                        "train_loss_average": [], 
                        "test_loss_average": []}
        


    def fit(self):

        iterator = tqdm(range(self.epochs), dynamic_ncols=True)

        self._do_batch_warmup()
        self._do_train_result_preallocation()
        self._do_test_result_preallocation()

        for epoch in iterator:
            
            train_loss_average = self._train_hook(epoch)

            # TODO: il test hook non lo voglio far partire nel training finale di submission, dove tutti i dati sono usati nel train. Per questo va impostato qualcosa nel config.
            test_loss_average = self._test_hook(epoch)
            

            # TODO: remember to save also the learning rate in tensorboard!
            self._do_metrics_computation(epoch, train_loss_average, test_loss_average)
            self._do_tensorboard_update(epoch, train_loss_average, test_loss_average)

            self._do_scheduler_step(train_loss_average)

            # Save the model state if the epoch is a multiple of the saving frequency  #TODO:sta cosa fa cagare
            if (epoch + 1) % self.model_saving_frequency == 0:
                os.makedirs(self.save_dir, exist_ok=True)
                torch.save(  #TODO: usare export visto che abbiamo creato l'interfaccia
                    self.model.state_dict(),
                    os.path.join(self.save_dir, self.model_name_prefix) + str(datetime.datetime.now())
                )
                
            self._do_progress_bar_step(iterator, epoch, train_loss_average, test_loss_average)
            
        return self.metrics





    def _do_batch_warmup(self):
        dataset = self.loader_train.dataset
        batch_size = dataConfig.batch_size

        batch = [dataset[i] for i in list(range(batch_size))]
        batch = default_collate(batch)

        inputs  = self.data_getter(batch).to(systemConfig.device)
        targets = self.target_getter(batch).to(systemConfig.device)

        model = self.model.eval()
        with torch.no_grad():
            preds = model(inputs)

        preds_to_store = preds.detach().cpu()
        targets_to_store = targets.detach().cpu()

        pred_tail_shape = tuple(preds_to_store.shape[1:]) 
        tgt_tail_shape  = tuple(targets_to_store.shape[1:])
        pred_dtype = preds_to_store.dtype
        tgt_dtype  = targets_to_store.dtype

        self.warmup_info = {
            "targets": targets,
            "preds" : preds,
            "pred_tail_shape": pred_tail_shape,
            "target_tail_shape": tgt_tail_shape,
            "pred_dtype": pred_dtype,
            "target_dtype": tgt_dtype,
        }

    def _do_train_result_preallocation(self):
        dataset_len = len(self.loader_train.dataset)

        pred_shape = (dataset_len, ) + self.warmup_info.get("pred_tail_shape")
        target_shape = (dataset_len, ) + self.warmup_info.get("target_tail_shape")

        self.train_pred = torch.empty(pred_shape, dtype=self.warmup_info.get("pred_dtype"))
        self.train_target = torch.empty(target_shape,  dtype=self.warmup_info.get("target_dtype"))


    def _do_test_result_preallocation(self):
        dataset_len = len(self.loader_test.dataset)

        pred_shape = (dataset_len, ) + self.warmup_info.get("pred_tail_shape")
        target_shape = (dataset_len, ) + self.warmup_info.get("target_tail_shape")

        self.test_pred = torch.empty(pred_shape, dtype=self.warmup_info.get("pred_dtype"))
        self.test_target = torch.empty(target_shape,  dtype=self.warmup_info.get("target_dtype"))


    def _train_hook(self, epoch):
        model = self.model.train()
        
        iterator = tqdm(self.loader_train, 
                        disable=not trainingConfig.progress_bar_on_batches_inside_epoch, 
                        dynamic_ncols=True)
        
        loss_average_tracker = AverageMeter()

        offset_saving_result = 0
        
        for i, sample in enumerate(iterator):
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Get inputs and targets from the sample, predict
            inputs = self.data_getter(sample).to(systemConfig.device)
            targets = self.target_getter(sample).to(systemConfig.device)
            predicts = model(inputs)
            
            # Calculate loss, backpropagate and update the weigths
            loss = self.loss_function(predicts, targets)
            loss.backward()
            self.optimizer.step()

            # Save the results
            preds_size = predicts.size(0)
            with torch.no_grad():
                self.train_pred[offset_saving_result : offset_saving_result + preds_size]   = predicts.detach().cpu()
                self.train_target[offset_saving_result : offset_saving_result + preds_size] = targets.detach().cpu()
            offset_saving_result += preds_size
            
            # Update loss average
            loss_average_tracker.update(loss.item())
            status = "{0}[Train][{1}] Loss_avg: {2:.5}, Loss: {3:.5}, LR: {4:.5}".format(
                "[{}/{}]".format(epoch, self.epochs), 
                i, 
                loss_average_tracker.avg, 
                loss_average_tracker.val, 
                self.optimizer.param_groups[0]["lr"]
            )
            
            # Update progress bar description
            iterator.set_description(status)
            
        return loss_average_tracker.avg
    

    def _test_hook(self, epoch): 
        
        model = self.model.eval()
        
        iterator = tqdm(self.loader_test,
                        disable=not trainingConfig.progress_bar_on_batches_inside_epoch, 
                        dynamic_ncols=True)
        
        loss_average_tracker = AverageMeter()
        
        offset_saving_result = 0
        
        for i, sample in enumerate(iterator):
            
            # Get inputs and targets from the sample
            inputs = self.data_getter(sample).to(systemConfig.device)
            targets = self.target_getter(sample).to(systemConfig.device)
        
            # Predict and calculate loss with no gradients
            with torch.no_grad():
                predicts = model(inputs)
                loss = self.loss_function(predicts, targets)

            # Save the results
            preds_size = predicts.size(0)
            with torch.no_grad():
                self.test_pred[offset_saving_result : offset_saving_result + preds_size]   = predicts.detach().cpu()
                self.test_target[offset_saving_result : offset_saving_result + preds_size] = targets.detach().cpu()
            offset_saving_result += preds_size
            
            # Update loss average and metric
            loss_average_tracker.update(loss.item())

            # If the model is classification, apply softmax to the predictions
            predicts = predicts.softmax(dim=1).detach() #TODO: make this softmax a configuration option

        
            # Update progress bar description
            status = "{0}[Test][{1}] Loss_avg: {2:.5}".format(
                "[{}/{}]".format(epoch, self.epochs), 
                i, 
                loss_average_tracker.avg
                )
            iterator.set_description(status)
        
        return loss_average_tracker.avg




    def _do_metrics_computation(self, epoch, train_loss_average, test_loss_average):
        self.metrics_train = metric_epoch_orchestrator(predictions=self.train_pred, targets=self.train_target)
        self.metrics_test = metric_epoch_orchestrator(predictions=self.test_pred, targets=self.test_target)    

        self.metrics['epoch'].append(epoch)
        self.metrics['metrics_train'].append(self.metrics_train)
        self.metrics['metrics_test'].append(self.metrics_test)
        self.metrics['train_loss_average'].append(train_loss_average)
        self.metrics['test_loss_average'].append(test_loss_average)
    
    def _do_tensorboard_update(self, epoch, train_loss_average, test_loss_average):
        self.tensorboard.add_training_metrics(metrics_train=self.metrics_train,
                                                metrics_test=self.metrics_test,
                                                epoch=epoch)
        
        self.tensorboard.add_losses(train_loss=train_loss_average, 
                                    test_loss=test_loss_average, 
                                    epoch = epoch)

        



    def _do_scheduler_step(self, train_loss_average):
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(train_loss_average)
            else:
                self.scheduler.step()

    def _do_progress_bar_step(self, iterator, epoch, train_loss_average, test_loss_average):
        iterator.set_description(
            "[{}/{}] Train Loss: {:.5f}, Test Loss: {:.5f}".format(
                epoch + 1, 
                self.epochs, 
                train_loss_average,
                test_loss_average
            )
        )