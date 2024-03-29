import torch
import torchvision.models as models

# download ResNet model
resnet101 = models.resnet101(pretrained=True)
