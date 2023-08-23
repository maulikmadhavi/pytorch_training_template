from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import logging
from torch.utils.tensorboard import SummaryWriter
import time
import copy
import torchmetrics  # To compute accuracy, F1 score, etc.
from tqdm import tqdm
import sys

sys.path.append("..")
from helpers.plotting import get_cam


def batch_forward(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """perform forward pass for a batch of inputs and labels
    Returns:
        tuple(torch.Tensor, torch.Tensor): outputs and loss
    """
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    return outputs, loss


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    epoch: int,
    tb_writer: SummaryWriter,
    num_classes: int,
) -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler, Dict[str, float]]:
    """Train for one epoch. Returns the model, optimizer, scheduler and accuracy for the epoch.

    Args:
        model (nn.Module): The model to train

    Returns:
        trained model, optimizer, scheduler, accuracy for the epoch
    """
    model.train()
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    total_loss = 0.0
    for batch_idx, (inputs, labels) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        outputs, loss = batch_forward(model, inputs, labels, criterion, device)
        _, preds = outputs.max(1)
        tb_writer.add_scalar(
            f"Step/loss-train",
            loss.item(),
            batch_idx + epoch * len(dataloader),
        )
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        metric.update(preds.detach().cpu(), labels.detach().cpu())
    scheduler.step()
    acc = {"train_acc": metric.compute(), "train_loss": total_loss / len(dataloader)}
    return model, optimizer, scheduler, acc


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
    tb_writer: SummaryWriter,
    num_classes: int,
) -> Dict[str, float]:
    """Validation for one epoch. Returns the accuracy and loss for the epoch.
    Returns:
    validation accuracy and loss for the epoch
    """
    model.eval()
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
    total_loss = 0.0
    for batch_idx, (inputs, labels) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        outputs, loss = batch_forward(model, inputs, labels, criterion, device)
        _, preds = outputs.max(1)
        tb_writer.add_scalar(
            f"Step/loss-val",
            loss.item(),
            batch_idx + epoch * len(dataloader),
        )
        total_loss += loss.item()
        metric.update(preds.detach().cpu(), labels.detach().cpu())
        f1.update(preds.detach().cpu(), labels.detach().cpu())
    acc = metric.compute()
    f1_score = f1.compute()
    return {
        "val_acc": acc,
        "val_f1": f1_score,
        "val_loss": total_loss / len(dataloader),
    }


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim,
    scheduler: lr_scheduler,
    num_epochs: int,
    logger: logging.Logger,
    tb_writer: SummaryWriter,
    num_classes: int,
) -> Tuple[nn.Module, float]:
    """Training model loop for num_epochs.
    Returns:
        tuple(nn.Module, float): best model and best accuracy
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch}/{num_epochs - 1}")
        logger.info("-" * 10)

        model, optimizer, scheduler, train_acc = train_one_epoch(
            model,
            dataloaders["train"],
            device,
            criterion,
            optimizer,
            scheduler,
            epoch,
            tb_writer,
            num_classes,
        )
        val_metrics = validate_one_epoch(
            model, dataloaders["val"], device, criterion, epoch, tb_writer, num_classes
        )

        model.train()
        cam_output, cam_input = get_cam(model, dataloaders["val"])
        tb_writer.add_image(
            "Step/cam",
            cam_output,
            epoch,
            dataformats="CHW",
        )
        tb_writer.add_image(
            "Step/cam_input",
            cam_input,
            epoch,
            dataformats="CHW",
        )

        model.to(device)
        logger.info(
            f"Train Loss: {train_acc['train_loss']:.4f}, Train Acc: {train_acc['train_acc']:.4f}"
        )
        logger.info(
            f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.4f}, Val F1: {val_metrics['val_f1']:.4f}"
        )

        #     # deep copy the model
    if val_metrics["val_acc"] > best_acc:
        best_acc = val_metrics["val_acc"]
        best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logger.info(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    logger.info(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc
