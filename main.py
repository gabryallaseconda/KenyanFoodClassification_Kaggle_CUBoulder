
import torch
import torch.nn as nn

from neural_network.configuration import system_setup, systemConfig, modelConfig

from neural_network.logging.tensorboard_interface import TensorBoardInterface
from neural_network.data.data_loader import get_data_loaders
from neural_network.model.model import get_model
from neural_network.training.optimizer import get_optimizer, get_scheduler
from neural_network.training.metrics import get_loss, AccuracyEstimator
from neural_network.training.trainer import Trainer

#from neural_network.trainer import Trainer

from neural_network.model.export import export_to_pth
from neural_network.inference.inference_on_submission import score_submission


system_setup()
print(f"Device: {systemConfig.device}")

tensorboard = TensorBoardInterface()
tensorboard.add_hyperparameters() 
tensorboard.add_configs()

train_loader, validation_loader, class_counts = get_data_loaders()
print(f"Distribution in classes: {class_counts}")

model = get_model(class_counts=class_counts)
print(f"Model: {modelConfig.model}")
tensorboard.add_model_graph(model)

optimizer = get_optimizer(parameters = model.parameters())
scheduler = get_scheduler(optimizer = optimizer)
loss = get_loss()
#metric = AccuracyEstimator()




if __name__ == "__main__":

    trainer = Trainer(
        model=model,

        loader_train=train_loader,
        loader_test=validation_loader,

        loss_function=loss,
        #metric_fn=metric,

        optimizer=optimizer,
        scheduler=scheduler,

        tensorboard=tensorboard,
    )
    
    metrics = trainer.fit()


# TODO: usare augmentation di neural_network dentro l'eda
# TODO: salvare (solo?) il modello che minimizza la loss del test set!


# Close TensorBoard writer

tensorboard.close()


export_to_pth(model)

score_submission(model)
