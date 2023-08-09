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

    def __init__(self, num_classes: int) -> None:
        """ Create a ResNet18 model with the given number of classes.

        Args:
            num_classes (int): number of classes in the dataset
        """
        super().__init__()
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the model.
        
        Args:
            x (Tensor): input tensor to the model of shape (N, C, H, W)
        
        Returns:
            Tensor: output tensor of shape (N, num_classes)
        """
        return self.model(x)


def test_model():
    """ Test the model """
    model = ResNet18(10)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)
    assert y.shape == (1, 10)


if __name__ == "__main__":
    test_model()