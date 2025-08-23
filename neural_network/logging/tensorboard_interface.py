
# Always remember:
# tensorboard --logdir runs

import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from neural_network.configuration import systemConfig, dataConfig, modelConfig, trainingConfig, metricsConfig, inferenceConfig


class TensorBoardInterface():
    def __init__(self):
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join("runs", modelConfig.model, run_id)
        self._writer = SummaryWriter(log_dir=log_dir)

    def add_model_graph(self, model):
        input_tensor = torch.randn(1, 3, dataConfig.resized_image_width, dataConfig.resized_image_height).to(systemConfig.device)
        self._writer.add_graph(model, input_tensor)

    def add_losses(self, train_loss, test_loss, epoch):
        if train_loss is not None:
            self._writer.add_scalar("train/loss", train_loss, epoch)
        if test_loss is not None:
            self._writer.add_scalar("test/loss", test_loss, epoch)

    def add_learning_rate(self, learning_rate, epoch):
        self._writer.add_scalar("operations/learning_rate", learning_rate, epoch)

    def add_training_metrics(self, metrics_train, metrics_test, epoch):
        metrics_names = metrics_train.keys()

        self._add_training_metrics_for_process(prefix="train/",
                                               metrics_names=metrics_names,
                                               metrics=metrics_train,
                                               epoch=epoch)

        if trainingConfig.run_test_process:
            if metrics_train.keys() != metrics_test.keys():
                raise KeyError("Keys in metrics dictionary must be the same for train and test!")
            
            self._add_training_metrics_for_process(prefix="test/",
                                                metrics_names=metrics_names,
                                                metrics=metrics_test,
                                                epoch=epoch)                   
        
        self._writer.flush()


    def _add_training_metrics_for_process(self, prefix, metrics_names, metrics, epoch):
        for metric_name in metrics_names:
            if metric_name in metricsConfig.type_scalar:
                self._writer.add_scalar(prefix+metric_name, 
                                        metrics.get(metric_name), 
                                        epoch)
            elif metric_name in metricsConfig.type_figure:
                self._writer.add_figure(prefix+metric_name, 
                                        metrics.get(metric_name), 
                                        epoch)
    
    def add_hyperparameters(self):
        hyperparameters = {
            "model": modelConfig.model,
            "head bias": modelConfig.set_heads_weights_bias_according_to_class_distribution,
            "optimizer": trainingConfig.optimizer,
            "learning rate": trainingConfig.learning_rate,
            "momentum": trainingConfig.momentum,
            "weight decay": trainingConfig.weight_decay,
            "scheduler": trainingConfig.scheduler,
            "scheduler step size": trainingConfig.scheduler_step_size,
            "scheduler gamma": trainingConfig.scheduler_gamma,
            "batch size": dataConfig.batch_size,
            "number of epochs": trainingConfig.number_of_epochs,
            "device": systemConfig.device,
            "seed1": systemConfig.seed,
            "seed2": dataConfig.seed,
            "test size": dataConfig.test_size,
        }

        # this is a workaround to avoid subfolder inside each run - tensorboard is very retarded
        def add_hparams_inline(writer, hparam_dict, metric_dict, global_step=0):
            from torch.utils.tensorboard.summary import hparams
            exp, ssi, sei = hparams(hparam_dict, metric_dict)
            fw = writer._get_file_writer()
            fw.add_summary(exp)
            fw.add_summary(ssi)
            fw.add_summary(sei)

        add_hparams_inline(self._writer, hyperparameters, {"_dummy": 0}, global_step=0)


    def add_configs(self):
        import json
        configs = {
            "system": self._obj_to_dict(systemConfig),
            "data":   self._obj_to_dict(dataConfig),
            "model":  self._obj_to_dict(modelConfig),
            "training": self._obj_to_dict(trainingConfig),
            "metrics": self._obj_to_dict(metricsConfig),
            "inference":self._obj_to_dict(inferenceConfig),
        }
        
        pretty = json.dumps(configs, indent=2, ensure_ascii=False, default=str)
        self._writer.add_text("hparams/configs", f"```json\n{pretty}\n```", global_step=0)


    def close(self):
        self._writer.close()


    @staticmethod
    def _obj_to_dict(obj):
        """
        Converte dataclass (istanze o classi), oggetti e mapping in un dict ricorsivo.
        - Dataclass istanza  -> asdict(instance)
        - Dataclass classe   -> {field: getattr(cls, field)} + (opzionale) attributi pubblici extra
        - Altri oggetti      -> __dict__ filtrato
        Converte tipi non serializzabili (Path, Enum, torch.device/dtype) in str.
        """
        import enum, pathlib
        from dataclasses import is_dataclass, asdict, fields
        import torch
        import inspect

        def convert(v):
            # primitivi
            if v is None or isinstance(v, (int, float, bool, str)):
                return v

            # mapping
            if isinstance(v, dict):
                return {str(k): convert(val) for k, val in v.items()}

            # sequenze
            if isinstance(v, (list, tuple, set)):
                return [convert(x) for x in v]

            # DATACLASS: istanza
            if is_dataclass(v) and not isinstance(v, type):
                return {k: convert(val) for k, val in asdict(v).items()}

            # DATACLASS: CLASSE (come nel tuo configuration.py)
            if isinstance(v, type) and is_dataclass(v):
                data = {}
                # 1) campi annotati (fields della dataclass)
                for f in fields(v):
                    data[f.name] = convert(getattr(v, f.name))
                # 2) opzionale: includi anche attributi pubblici non-callable non-privati
                for name, val in vars(v).items():
                    if name.startswith("_"):
                        continue
                    if callable(val):
                        continue
                    if name in data:  # già preso dai fields
                        continue
                    data[name] = convert(val)
                return data

            # oggetti generici: usa __dict__ pulito se disponibile
            if hasattr(v, "__dict__"):
                return {k: convert(val) for k, val in v.__dict__.items()
                        if not k.startswith("_") and not callable(val)}

            # tipi speciali -> stringa
            if isinstance(v, (enum.Enum, pathlib.Path)):
                return str(v)
            try:
                if isinstance(v, (torch.device, torch.dtype)):
                    return str(v)
            except Exception:
                pass

            # fallback
            return str(v)

        return convert(obj)
    
