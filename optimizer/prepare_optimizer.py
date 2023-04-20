"""
This script is used to prepare the optimizer function for training.
"""
# Standard library imports
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def prepare_optimizer(
    model: nn.Module,
    lr: float,
    wd: float,
    momentum: float,
    sch_step: int,
    sch_gamma: float,
) -> Tuple[optim.SGD, optim.lr_scheduler.StepLR]:
    """
    This function prepares the optimizer for training.
    """
    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

    # Define the scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=sch_step, gamma=sch_gamma
    )

    return optimizer, scheduler


# def prepare_optimizer(
#     model, lr, wd, momentum, sch_step, sch_gamma, optimizer_name, scheduler_type,
# ):
#     """
#     This function is used to get the loss function from available list of pytorch loss functions.
#     """
#     return getattr(torch.nn, loss_name)
