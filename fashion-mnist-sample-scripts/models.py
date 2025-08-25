from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class TorchBaseModel(nn.Module, ABC):
    def __init__(self, ):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """Forward pass must be implemented by all models"""
        pass

    def save(self, save_path: str):
        torch.save(self.state_dict(), save_path)

    def load(self, load_path: str, map_location: str = "cpu"):
        """
        load_state_dict does not return a model — 
        it returns a NamedTuple with missing/unexpected keys (kind of a status dict). 
        The actual state is loaded into self in place.
        Basically this TorchBaseModel itself is the self.model
        That is why you just directly do self.eval because "self" is the model itself
        """
        self.load_state_dict(torch.load(load_path), map_location=map_location)
        self.eval()

    
class CNNCustomModel(TorchBaseModel):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Expected input shape: # batch, channels, H, W
        """
        x = self.features(x)
        x = torch.flatten(x, 1) # flatten all except batch
        x = self.classifier(x)
        return x
