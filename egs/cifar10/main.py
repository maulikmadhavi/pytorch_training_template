"""
Pytorch model training templates
"""
from typing import List, Tuple
import time
import os
import copy

import torch
import torchvision

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
import logging
from ptt.loss.prepare_loss import get_loss
from ptt.optimizer.prepare_optimizer import prepare_optimizer
from ptt.data.get_transforms import train_transforms, val_transforms, inv_normalize
from ptt.data.prepare_data import get_data
from ptt.data.get_loader import get_loader
from ptt.trainer.base_trainer import train_model
from ptt.models.prepare_model import ResNet18
import hydra
from omegaconf import DictConfig, OmegaConf

Tensor = torch.tensor
cudnn.benchmark = True


# ======================= Hyper parameters ================================
# BS = 32
# LR = 0.01
# NEPOCH = 15
# NB = 20
# WD = 1e-4
# MOMENTUM = 0.9
# SCH_STEP = 10
# SCH_GAMMA = 0.1
# LOGFILE = "logs/log.txt"
# TBLOGS = "tblogs/"


# Moving alll the hyperparameters to hydra config file
@hydra.main(
    config_path="/home/maulik/Documents/Tool/pytorch_training_template/egs/cifar10/",
    config_name="cifar10.yaml",
)
def main(cfg: DictConfig) -> None:
    # ======================== Hyper parameters ==============================
    BS = cfg.data.BS
    LR = cfg.train.LR
    NEPOCH = cfg.train.NEPOCH
    WD = cfg.train.WD
    MOMENTUM = cfg.train.MOMENTUM
    SCH_STEP = cfg.train.SCH_STEP
    SCH_GAMMA = cfg.train.SCH_GAMMA
    LOGFILE = cfg.output.LOGFILE
    TBLOGS = cfg.output.TBLOGS

    # ======================== Data preparation ==============================
    # Set the device to GPU if it exists, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset, valset = get_data(train_transforms, val_transforms)

    # Create a dataloader for the training data
    train_loader, val_loader = get_loader(trainset, valset, batch_size=BS)

    # Create a dictionary of the datasets and their sizes
    image_datasets = {"train": trainset, "val": valset}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    # Create a dictionary of the classes
    class_names = image_datasets["train"].classes

    # Create a dictionary of the dataloaders
    dataloaders = {"train": train_loader, "val": val_loader}

    # ================== Model defination ===============================
    # Define the number of classes
    classes = class_names

    # Move the model to the GPU if it is available
    model = ResNet18(num_classes=len(classes))
    model = model.to(device)

    # ================= Optimizer and loss function =================
    criterion = get_loss("CrossEntropyLoss")
    optimizer, exp_lr_scheduler = prepare_optimizer(
        model, LR, WD, MOMENTUM, SCH_STEP, SCH_GAMMA
    )

    # ========================== MAIN SCRIPT =====================================

    # Create tensorboard logger
    tb_writer = SummaryWriter(TBLOGS)
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images.to(device))
    tb_writer.add_image("image grid visulization", inv_normalize(img_grid))
    tb_writer.add_graph(model, images.to(device))

    # Create file logger
    if not os.path.exists(os.path.dirname(LOGFILE)):
        os.makedirs(os.path.dirname(LOGFILE))

    # Create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    log_file = LOGFILE
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add the formatter to the handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Started Training")
    # Call the train_model function
    model, best_acc = train_model(
        model,
        dataloaders,
        device,
        criterion,
        optimizer,
        exp_lr_scheduler,
        NEPOCH,
        logger,
        tb_writer,
        num_classes=len(classes),
    )

    torch.save(model, f"best_model_{best_acc:0.3f}.pth")

    logger.info("Finished Training")
    tb_writer.close()


if __name__ == "__main__":
    main()
