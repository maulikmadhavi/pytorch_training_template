"""
This script is used to prepare the dataloader for training and validation.
"""
from typing import Tuple
import torch.utils.data as data


def get_loader(
    train_data: data.Dataset,
    val_data: data.Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[data.DataLoader, data.DataLoader]:
    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
