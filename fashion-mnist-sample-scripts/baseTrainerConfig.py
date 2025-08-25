from typing import Optional
from pydantic import BaseModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader            
from torch.optim.lr_scheduler import _LRScheduler


class BaseTrainerConfig(BaseModel):
    model: nn.Module
    device: torch.device
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    train_dataloader: DataLoader
    val_dataloader: Optional[DataLoader] = None
    epochs: int = 10
    lr_scheduler: Optional[_LRScheduler] = None


    class Config:
        arbitrary_types_allowed = True