import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model