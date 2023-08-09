"""
This script is used to prepare the loss function for training.
"""

# Standard library imports
import torch.nn as nn

# Cross E
def get_loss(loss_name: str) -> nn.Module:
    """
    This function is used to get the loss function from available list of pytorch loss functions.    
    """
    return getattr(nn, loss_name)(reduction="mean")
    