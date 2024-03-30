import torch
import torchvision.models as models

# download ResNet model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

