import torch
import torchvision.models as models

# download ResNet model
resnet101 = models.resnet101(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
