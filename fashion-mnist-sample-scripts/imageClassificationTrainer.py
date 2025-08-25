from typing import Any, Dict
import torch
from tqdm import tqdm
from baseTrainer import BaseTrainer
from torch.utils.data import DataLoader

class ImageClassificationTrainer(BaseTrainer):

    def _train_one_epoch(self) -> Dict[str, Any]:
        self.model.train()
        total_loss = 0. # cumulative loss across all images in 1 entire epoch
        correct_preds, total_num_images = 0, 0

        for batch in tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch} Training"):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            # -- MODEL TRAINING AND WEIGHT UPDATES --
            self.optimizer.zero_grad()
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels) # this is avg loss across entire batch
            batch_loss.backward()
            self.optimizer.step()
            # TO DO: integrate with LR Scheduler

            # -- TRAINING STATS --
            total_loss += batch_loss.item() * images.size(0) # get total loss for entire batch = avg_loss * num_images, because last batch might be diff number
            _, predicted = torch.max(outputs.data, 1)
            total_num_images += labels.size(0)
            correct_preds += (predicted==labels).sum().item()

        avg_epoch_loss = total_loss / total_num_images
        epoch_accuracy = correct_preds / total_num_images

        return {"loss": avg_epoch_loss, "accuracy": epoch_accuracy}
    
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0. # cumulative loss across all images in 1 entire epoch
        correct_preds, total_num_images = 0, 0

        for batch in tqdm(dataloader):
            images, labels = batch
            images, labels = images.to(self.device), labels.to(self.device)

            # -- MODEL VALIDATION --
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels) # this is avg loss across entire batch

            # -- VALIDATION STATS --
            total_loss += batch_loss.item() * images.size(0) # get total loss for entire batch = avg_loss * num_images, because last batch might be diff number
            _, predicted = torch.max(outputs.data, 1)
            total_num_images += labels.size(0)
            correct_preds += (predicted==labels).sum().item()

        avg_epoch_loss = total_loss / total_num_images
        epoch_accuracy = correct_preds / total_num_images

        return {"loss": avg_epoch_loss, "accuracy": epoch_accuracy}
    
    def _validate(self) -> Dict[str, Any]:
        return self._evaluate(self.val_dataloader)

    def test(self) -> Dict[str, Any]:
        test_stats = self._evaluate(self.test_dataloader)
        test_loss, test_acc = test_stats.get("loss"), test_stats.get("accuracy")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        return test_stats