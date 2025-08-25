from abc import abstractmethod, ABC
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader            
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from baseTrainerConfig import BaseTrainerConfig

class BaseTrainer(ABC):
    def __init__(self, config: BaseTrainerConfig):
        self.config = config 
        self.device = self.config.device
        self.model = self.config.model.to(self.device)
        self.optimizer = self.config.optimizer
        self.criterion = self.config.criterion
        self.train_dataloader = self.config.train_dataloader
        self.val_dataloader = self.config.val_dataloader
        self.epochs = self.config.epochs
        if not self.config.lr_scheduler:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)
        else:
            self.lr_scheduler = self.config.lr_scheduler

        # FOR TRACKING TRAINING STATISTICS
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float("inf")
        self.best_model_state = None

    def train(self) -> None:
        for _ in range(self.epochs):
            self.current_epoch += 1

            # --- TRAIN ---
            train_stats = self._train_one_epoch()
            if not isinstance(train_stats, dict):
                raise ValueError("_train_one_epoch must return a dict of metrics")
            train_loss, train_acc = train_stats.get("loss"), train_stats.get("accuracy")
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # --- VALIDATION ---
            if self.val_dataloader:
                val_stats = self._validate()
                if not isinstance(val_stats, dict):
                    raise ValueError("_validate must return a dict of metrics")
                val_loss, val_acc = val_stats.get("loss"), val_stats.get("accuracy")
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss 
                    self.best_model_state = self.model.state_dict()
                # TO DO: save model checkpoint or best model state

            val_loss_str = f"{val_loss:.4f}" if self.val_dataloader else "N/A"
            val_acc_str = f"{val_acc:.4f}" if self.val_dataloader else "N/A"
            print(f"Epoch {self.current_epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss_str}, val_acc={val_acc_str}")

    @abstractmethod
    def _train_one_epoch(self) -> Dict[str, Any]:
        """Return a dict of metrics, e.g., {'loss': ..., 'accuracy': ..., 'f1': ...}"""
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Return a dict of metrics, e.g., {'loss': ..., 'accuracy': ..., 'f1': ...}"""
        raise NotImplementedError
    
    @abstractmethod
    def _validate(self) -> Dict[str, Any]:
        """Return a dict of metrics, e.g., {'loss': ..., 'accuracy': ..., 'f1': ...}"""
        raise NotImplementedError

    @abstractmethod
    def test(self) -> Dict[str, Any]:
        """Return a dict of metrics on the test set."""
        raise NotImplementedError
    

# TO DO
# Add features only when you actually need them, e.g.:
# Checkpoint saving
# Early stopping
# Mixed precision / gradient clipping
# Logging to CSV / TensorBoard
# Multi-task support