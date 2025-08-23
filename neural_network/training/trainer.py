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
        
        self.data_getter = lambda sample: sample["image"]
        self.target_getter = lambda sample: torch.tensor(sample["target"]) #TODO check this as it rise a UserWarning, recommendig to use .clone() for copy construct a tensor
        self.get_key_metric = lambda metric: metric["top1"]
        

    def fit(self):

        iterator = tqdm(range(self.epochs), 
                        dynamic_ncols=True)

        self._do_batch_warmup()
        self._do_results_preallocation()

        for epoch in iterator:
            train_loss_average = self._train_process(epoch)

            if trainingConfig.run_test_process:
                test_loss_average = self._test_process(epoch) # TODO: il test process non lo voglio far partire nel training finale di submission, dove tutti i dati sono usati nel train. Per questo va impostato qualcosa nel config.
            else:
                test_loss_average = self._dummy_when_not_run_test_process()

            self._do_metrics_computation(epoch, train_loss_average, test_loss_average)
            self._do_tensorboard_update(epoch, train_loss_average, test_loss_average)
            self._do_scheduler_step(train_loss_average)
            self._do_model_saving(epoch)
            self._do_progress_bar_step(iterator, epoch, train_loss_average, test_loss_average)
            
        return


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


    def _do_results_preallocation(self):
        pred_shape, target_shape = self._compute_result_shapes(self.loader_train)
        self.train_pred = torch.empty(pred_shape, dtype=self.warmup_info.get("pred_dtype"))
        self.train_target = torch.empty(target_shape,  dtype=self.warmup_info.get("target_dtype"))

        if trainingConfig.run_test_process:
            pred_shape, target_shape = self._compute_result_shapes(self.loader_test)
            self.test_pred = torch.empty(pred_shape, dtype=self.warmup_info.get("pred_dtype"))
            self.test_target = torch.empty(target_shape,  dtype=self.warmup_info.get("target_dtype"))

    def _compute_result_shapes(self, data_loader):
        dataset_len = len(data_loader.dataset)
        pred_shape = (dataset_len, ) + self.warmup_info.get("pred_tail_shape")
        target_shape = (dataset_len, ) + self.warmup_info.get("target_tail_shape")
        return pred_shape, target_shape


    def _train_process(self, epoch):
        model = self.model.train()
        
        iterator = tqdm(self.loader_train, 
                        disable=not trainingConfig.progress_bar_on_batches_inside_epoch, 
                        dynamic_ncols=True)
        
        loss_average_tracker = AverageMeter()

        offset_saving_result = 0
        
        for i, sample in enumerate(iterator):
            self.optimizer.zero_grad()                                              # Reset gradients
            
            inputs = self.data_getter(sample).to(systemConfig.device)               # Get inputs and targets from the sample, predict
            targets = self.target_getter(sample).to(systemConfig.device)
            predicts = model(inputs)
            
            loss = self.loss_function(predicts, targets)                            # Calculate loss, backpropagate and update the weigths
            loss.backward()
            self.optimizer.step()
            loss_average_tracker.update(loss.item())

            preds_size = predicts.size(0)                                           # Save the results
            with torch.no_grad():
                self.train_pred[offset_saving_result : offset_saving_result + preds_size]   = predicts.detach().cpu()
                self.train_target[offset_saving_result : offset_saving_result + preds_size] = targets.detach().cpu()
            offset_saving_result += preds_size
                        
            status = "[{0}/{1}][Train][{2}] loss: {3:.5}, lr: {4:.5}".format(               # Update progress bar description
                epoch, self.epochs, i, 
                loss_average_tracker.avg, 
                self.optimizer.param_groups[0]["lr"])
            iterator.set_description(status)
            
        return loss_average_tracker.avg
    

    def _test_process(self, epoch): 
        model = self.model.eval()
        
        iterator = tqdm(self.loader_test,
                        disable=not trainingConfig.progress_bar_on_batches_inside_epoch, 
                        dynamic_ncols=True)
        
        loss_average_tracker = AverageMeter()
        
        offset_saving_result = 0
        
        for i, sample in enumerate(iterator):            
            inputs = self.data_getter(sample).to(systemConfig.device)               # Get inputs and targets from the sample
            targets = self.target_getter(sample).to(systemConfig.device)
            
            with torch.no_grad():                                                   # Predict and calculate loss with no gradients
                predicts = model(inputs)
                loss = self.loss_function(predicts, targets)
            loss_average_tracker.update(loss.item())

            preds_size = predicts.size(0)                                           # Save the results
            with torch.no_grad():
                self.test_pred[offset_saving_result : offset_saving_result + preds_size]   = predicts.detach().cpu()
                self.test_target[offset_saving_result : offset_saving_result + preds_size] = targets.detach().cpu()
            offset_saving_result += preds_size
            
            status = "[{0}/{1}][Test][{2}] loss: {3:.5}".format(                      # Update progress bar description
                epoch, self.epochs, i, 
                loss_average_tracker.avg)
            iterator.set_description(status)
        
        return loss_average_tracker.avg


    def _dummy_when_not_run_test_process(self):
        self.test_pred = None
        self.test_target = None
        return 0.0
    

    def _do_metrics_computation(self, epoch, train_loss_average, test_loss_average):
        self.metrics_train = metric_epoch_orchestrator(predictions=self.train_pred, targets=self.train_target)
        if trainingConfig.run_test_process:
            self.metrics_test = metric_epoch_orchestrator(predictions=self.test_pred, targets=self.test_target)    
        else:
            self.metrics_test = None

    def _do_tensorboard_update(self, epoch, train_loss_average, test_loss_average):
        self.tensorboard.add_training_metrics(metrics_train=self.metrics_train,
                                                metrics_test=self.metrics_test,
                                                epoch=epoch)
        
        self.tensorboard.add_losses(train_loss=train_loss_average, 
                                    test_loss=test_loss_average, 
                                    epoch = epoch)
        
        self.tensorboard.add_learning_rate(learning_rate=self.optimizer.param_groups[0]['lr'],
                                           epoch=epoch)


    def _do_scheduler_step(self, train_loss_average):
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(train_loss_average)
            else:
                self.scheduler.step()


    def _do_model_saving(self, epoch):
            if (epoch + 1) % trainingConfig.model_saving_frequency == 0:
                os.makedirs(trainingConfig.model_dir, exist_ok=True)
                
                filename = trainingConfig.model_name_prefix + str(datetime.datetime.now())
                filepath = os.path.join(trainingConfig.model_dir,filename)
                
                torch.save(  #TODO: usare export visto che abbiamo creato l'interfaccia
                    self.model.state_dict(),
                    filepath
                )


    def _do_progress_bar_step(self, iterator, epoch, train_loss_average, test_loss_average):
        iterator.set_description(
            "[{0}/{1}][End of epoch] train loss: {2:.5f}, test loss: {3:.5f}".format(
                epoch + 1, 
                self.epochs, 
                train_loss_average,
                test_loss_average)
        )