"""
Pytorch model training templates
"""
import time
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch.optim as optim


cudnn.benchmark = True


# ======================= Hyper parameters ================================
BS = 32
LR = 0.01
NEPOCH = 20
NB = 20
WD = 1e-4
MOMENTUM = 0.9
SCH_STEP = 10
SCH_GAMMA = 0.1

# ======================== Data preparation ==============================
train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=False, transform=train_transforms
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BS, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="../data", train=False, download=False, transform=val_transforms
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BS, shuffle=False, num_workers=2
)

image_datasets = {"train": trainset, "val": testset}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes
dataloaders = {"train": trainloader, "val": testloader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================== Model defination ===============================
classes = class_names

model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)
# ================= Optimizer and loss function =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=SCH_STEP, gamma=SCH_GAMMA)

# ================= Training loops ===============================
def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim,
    scheduler: lr_scheduler,
    num_epochs: int = 2,
) -> nn.Module:
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model = train_model(model, criterion, optimizer, exp_lr_scheduler, NEPOCH)
print("Finished Training")
