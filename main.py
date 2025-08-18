
import torch
import torch.nn as nn

from neural_network.configuration import system_setup, systemConfig, modelConfig

from neural_network.logging.visualizer import TensorBoardVisualizer
from neural_network.data.data_loader import get_data_loaders
from neural_network.model.model import get_model
from neural_network.training.optimizer import get_optimizer, get_scheduler
from neural_network.training.measures import get_loss, AccuracyEstimator


from neural_network.trainer import Trainer

from neural_network.model.export import export_to_pth
from neural_network.inference.inference_on_submission import score_submission


system_setup()
print(f"Device: {systemConfig.device}")

visualizer = TensorBoardVisualizer()

train_loader, validation_loader, class_counts = get_data_loaders()
print(f"Distribution in classes: {class_counts}")

model = get_model(class_counts=class_counts)
print(f"Model: {modelConfig.model}")
visualizer.add_model_graph(model)

optimizer = get_optimizer(parameters = model.parameters())
scheduler = get_scheduler(optimizer = optimizer)
loss = get_loss()
metric = AccuracyEstimator(topk=(1,))




if __name__ == "__main__":

    trainer = Trainer(
        model=model,

        loader_train=train_loader,
        loader_test=validation_loader,

        loss_fn=loss,
        metric_fn=metric,

        optimizer=optimizer,
        scheduler=scheduler,

        visualizer=visualizer,
    )
    
    metrics = trainer.fit()




# Close TensorBoard writer
visualizer.close_tensorboard()


export_to_pth(model)

score_submission(model)

