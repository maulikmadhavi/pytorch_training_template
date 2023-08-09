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
    use_scheduler: bool = True,
) -> Tuple[optim.SGD, optim.lr_scheduler.StepLR]:
    """
    This function prepares the optimizer for training.
    """
    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

    # Define the scheduler
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=sch_step, gamma=sch_gamma
        )
    else:
        scheduler = None
    return optimizer, scheduler


# def prepare_optimizer(
#     model, lr, wd, momentum, sch_step, sch_gamma, optimizer_name, scheduler_type,
# ):
#     """
#     This function is used to get the loss function from available list of pytorch loss functions.
#     """
#     return getattr(torch.nn, loss_name)

def test_prepare_optimizer():
    """
    This function tests the prepare_optimizer function.
    """
    model = nn.Linear(10, 2)
    lr = 0.01
    wd = 1e-4
    momentum = 0.9
    sch_step = 10
    sch_gamma = 0.1
    optimizer, scheduler = prepare_optimizer(
        model, lr, wd, momentum, sch_step, sch_gamma
    )
    assert isinstance(optimizer, optim.SGD)
    assert isinstance(scheduler, optim.lr_scheduler.StepLR)
    
if __name__ == "__main__":
    test_prepare_optimizer()