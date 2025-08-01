

import torchvision.models as models
import torch.nn as nn


import torch




# class SimpleCNNModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 3, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128, num_classes)
#         )
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
    


# def get_model(num_classes):
#     """
#     Returns a model with the specified number of output classes.
    
#     Args:
#         num_classes (int): Number of output classes for the model.
        
#     Returns:
#         nn.Module: A PyTorch model with the specified number of output classes.
#     """
#     # Use the SimpleCNNModel defined above
#     return SimpleCNNModel(num_classes=num_classes)





def get_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    trainable_layers: int = 1
) -> nn.Module:
    """
    Returns a ResNet-50 model configured for transfer learning or fine-tuning.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): If True, load ImageNet pretrained weights.
        freeze_backbone (bool): If True, freeze backbone layers before fine-tuning.
        trainable_layers (int): Number of final ResNet blocks to leave trainable (1-4).

    Returns:
        nn.Module: ResNet-50 model.
    """
    # 1. Load ResNet-50
    model = models.resnet50(pretrained=pretrained)
    model = models.resnet101(pretrained=pretrained).to(device="cuda" if torch.cuda.is_available() else "cpu")

    # 2. Optionally freeze backbone
    if freeze_backbone:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Define block names in order
        blocks = ['layer1', 'layer2', 'layer3', 'layer4']
        # Clamp trainable_layers
        trainable_layers = max(0, min(trainable_layers, len(blocks)))
        # Unfreeze last `trainable_layers` blocks
        for block_name in blocks[-trainable_layers:]:
            block = getattr(model, block_name)
            for param in block.parameters():
                param.requires_grad = True

    # 3. Replace the final fully-connected layer
    in_features = model.fc.in_features  # typically 2048
    model.fc = nn.Linear(in_features, num_classes)
    # Ensure the new fc is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
