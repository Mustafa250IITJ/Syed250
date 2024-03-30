import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
# download ResNet model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

# save the model to a folder
torch.save(resnet50.state_dict(), 'resnet50.pth')
