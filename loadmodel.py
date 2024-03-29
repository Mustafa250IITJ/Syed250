import torch
import torchvision.models as models

# download ResNet model
# resnet101 = models.resnet101(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

# save the model to a folder
torch.save(resnet50.state_dict(), 'resnet50.pth')
