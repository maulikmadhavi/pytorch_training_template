"""
This script prepares the data for the model.
"""
from typing import Tuple
import torchvision


def get_data(
    train_transforms, val_transforms
) -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    # Load the training data and apply the training transforms
    trainset = torchvision.datasets.CIFAR10(
        root="/home/maulik/practice/pytorch_templates/cifar10_data",
        train=True,
        download=False,
        transform=train_transforms,
    )

    # Load the test data and apply the validation transforms
    testset = torchvision.datasets.CIFAR10(
        root="/home/maulik/practice/pytorch_templates/cifar10_data",
        train=False,
        download=False,
        transform=val_transforms,
    )
    return trainset, testset
