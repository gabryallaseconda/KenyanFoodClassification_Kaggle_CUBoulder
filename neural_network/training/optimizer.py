import torch.optim as optim

from neural_network.configuration import trainingConfig


def get_optimizer(parameters):

    if trainingConfig.optimizer == "adam":
        return optim.Adam(params=parameters, 
                          lr=trainingConfig.learning_rate, 
                          weight_decay=trainingConfig.weight_decay)

    elif trainingConfig.optimizer == "adamw":
        return optim.AdamW(params=parameters, 
                           lr=trainingConfig.learning_rate, 
                           weight_decay=trainingConfig.weight_decay)

    elif trainingConfig.optimizer == "sgd":
        return optim.SGD(params=parameters, 
                         lr=trainingConfig.learning_rate, 
                         momentum=trainingConfig.momentum, 
                         weight_decay=trainingConfig.weight_decay)

    else: raise ValueError(f"Unsupported optimizer: {trainingConfig.optimizer}")

    
def get_scheduler(optimizer):
    if trainingConfig.scheduler == "step_lr":
        return optim.lr_scheduler.StepLR(optimizer, 
                                         step_size=trainingConfig.scheduler_step_size, 
                                         gamma=trainingConfig.scheduler_gamma)


