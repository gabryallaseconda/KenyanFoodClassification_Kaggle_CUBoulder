

import torchvision.models as models
import torch.nn as nn


import torch

import torch.nn.functional as F
import numpy as np


from neural_network.configuration import modelConfig, systemConfig

def get_model(class_counts=None):
    model = model_registry[modelConfig.model](num_classes=len(class_counts))
    _set_heads_weights_bias_according_to_class_distribution(model, class_counts)
    model.to(systemConfig.device)
    return model


def _set_heads_weights_bias_according_to_class_distribution(model, class_count):
    if modelConfig.set_heads_weights_bias_according_to_class_distribution:
        total = sum(class_count.values())
        class_count_proportion = [class_count.get(i, 1) / total 
                                  for i in range(len(class_count))]
        class_probs = torch.tensor(class_count_proportion,
                                   dtype=torch.float32)
        bias = torch.log(class_probs)
        
        with torch.no_grad():
            model.classifier[-1].bias.copy_(bias)


class SimpleCNNModel(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

class ModestCNNModel(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((10, 10)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*10*10, 64*10*10),

            nn.ReLU(),
            nn.Linear(64*10*10, 8*10*10),

            nn.ReLU(),
            nn.Linear(8*10*10, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

class LargeCNNModel(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.max_pool = nn.AdaptiveMaxPool2d((2, 2))
        
        self.classifier = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(1024*(5*5 + 2*2), 1024*10),
            nn.ReLU(),

            nn.Linear(1024*10, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.features(x)

        avg_pooled = self.avg_pool(features)
        avg_pooled = avg_pooled.flatten(1)
        max_pooled = self.max_pool(features)
        max_pooled = max_pooled.flatten(1)

        concat = torch.cat((avg_pooled, max_pooled), dim=1)

        y = self.classifier(concat)

        return y


model_registry = {
    "simple_cnn": SimpleCNNModel,
    "modest_cnn": ModestCNNModel,
    "large_cnn": LargeCNNModel,
    "resnet50": lambda num_classes: models.resnet50(pretrained=True, num_classes=num_classes)
    }




# def get_model(
#     num_classes: int,
#     pretrained: bool = True,
#     freeze_backbone: bool = True,
#     trainable_layers: int = 1
# ) -> nn.Module:
#     """
#     Returns a ResNet-50 model configured for transfer learning or fine-tuning.

#     Args:
#         num_classes (int): Number of output classes.
#         pretrained (bool): If True, load ImageNet pretrained weights.
#         freeze_backbone (bool): If True, freeze backbone layers before fine-tuning.
#         trainable_layers (int): Number of final ResNet blocks to leave trainable (1-4).

#     Returns:
#         nn.Module: ResNet-50 model.
#     """
#     # 1. Load ResNet-50
#     #model = models.resnet50(pretrained=pretrained)
#     model = models.resnet101(pretrained=pretrained).to(device="cuda" if torch.cuda.is_available() else "cpu")

#     # 2. Optionally freeze backbone
#     if freeze_backbone:
#         # Freeze all parameters
#         for param in model.parameters():
#             param.requires_grad = False

#         # Define block names in order
#         blocks = ['layer1', 'layer2', 'layer3', 'layer4']
#         # Clamp trainable_layers
#         trainable_layers = max(0, min(trainable_layers, len(blocks)))
#         # Unfreeze last `trainable_layers` blocks
#         for block_name in blocks[-trainable_layers:]:
#             block = getattr(model, block_name)
#             for param in block.parameters():
#                 param.requires_grad = True

#     # 3. Replace the final fully-connected layer
#     in_features = model.fc.in_features  # typically 2048
#     model.fc = nn.Linear(in_features, num_classes)
#     # Ensure the new fc is trainable
#     for param in model.fc.parameters():
#         param.requires_grad = True

#     return model
