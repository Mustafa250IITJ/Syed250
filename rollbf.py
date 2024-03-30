import torch
import torchvision.models as models

# download ResNet model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

# save the model to a folder
torch.save(resnet50.state_dict(), 'resnet50.pth')
