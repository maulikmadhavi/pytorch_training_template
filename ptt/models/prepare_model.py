import torch
import torch.nn as nn
import torchvision

Tensor = torch.tensor


# class ResNet18(nn.Module):
#     "Modify to append CAM"

#     def __init__(self, num_classes):
#         super().__init__()
#         model = torchvision.models.resnet18(pretrained=True)
#         self.fc = nn.Linear(model.fc.in_features, num_classes)
#         model.fc = nn.Identity()
#         self.model = model
#         # placeholder for the gradients
#         self.gradients = None

#     # hook for the gradients of the activations
#     def activations_hook(self, grad):
#         self.gradients = grad

#     def forward(self, x: Tensor) -> Tensor:
#         # See note [TorchScript super()]
#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)

#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)

#         # register the hook
#         # if self.training:
#         #     h = x.register_hook(self.activations_hook)

#         x = self.model.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

#     # method for the gradient extraction
#     def get_activations_gradient(self):
#         return self.gradients

#     # method for the activation exctraction
#     def get_activations(self, x):
#         return self.features_conv(x)


class ResNet18(nn.Module):
    "Modify to append CAM"

    def __init__(self, num_classes):
        super().__init__()
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
