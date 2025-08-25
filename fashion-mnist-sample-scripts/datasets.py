import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import pandas as pd
from torchvision import transforms

class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """

    @abstractmethod
    def __len__(self):
        """Return the number of samples in the dataset"""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """
        Return a single sample at index `idx`.
        Expected to return:
        - features (tensor)
        - label (tensor)
        """
        pass



class FashionMnistDataset(BaseDataset):
    def __init__(self, data_path, split_type):
        super().__init__()
        self.data_path = data_path
        self.split_type = split_type
        self.data = pd.read_csv(data_path)
        self.images = self.data.iloc[:,1:].to_numpy(dtype="float32")
        self.labels = self.data.iloc[:,:1].to_numpy(dtype="int64").squeeze()
        self.images = self.images.reshape(-1, 28, 28)

        # Define transforms
        self.train_transforms = transforms.Compose([
            # transforms.ToPILImage(),                     # Convert numpy array to PIL image
            # transforms.RandomRotation(degrees=15),      # Small rotation for augmentation
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random shift
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),                       # Convert PIL image to tensor
            # transforms.Normalize((0.5,), (0.5,))        # Normalize grayscale to [-1,1]
        ])
        self.val_test_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image /= 255
        if self.split_type == "train":
            image = self.train_transforms(image)
        else:
            image = self.val_test_transforms(image)
        label = torch.tensor(label)
        return image, label

