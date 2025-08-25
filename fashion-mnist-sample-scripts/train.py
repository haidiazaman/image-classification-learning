import torch
from models import CNNCustomModel
from datasets import FashionMnistDataset
from torch.utils.data import DataLoader
from imageClassificationTrainer import ImageClassificationTrainer
from baseTrainerConfig import BaseTrainerConfig
from torch.optim import Adam
import torch.nn as nn

# set for reproducibility
torch.manual_seed(42)

def main(config):
    trainer = ImageClassificationTrainer(config=config)
    trainer.train()

if __name__ == "__main__":
    trainPath = 'data/fashionmnist/fashion-mnist_train_split.csv'
    valPath = 'data/fashionmnist/fashion-mnist_val_split.csv'
    learning_rate = 1e-3 # 1e-3=0.001
    batch_size = 128
    epochs = 10

    trainData = FashionMnistDataset(trainPath, "train")
    valData = FashionMnistDataset(valPath, "val")
    train_loader = DataLoader(trainData, batch_size=batch_size, pin_memory=True, shuffle=True)
    val_loader = DataLoader(valData, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = CNNCustomModel(num_classes=10)

    config = BaseTrainerConfig(
        model=model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        optimizer=Adam(model.parameters(), lr=learning_rate),
        criterion=nn.CrossEntropyLoss(), train_dataloader=train_loader, val_dataloader=val_loader,
        epochs=epochs
    )
    main(config)