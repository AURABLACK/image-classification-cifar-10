from logging import root
import hydra

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from imageclassification import models
from imageclassification import dataset
from imageclassification import utils


@hydra.main(config_path="./configs", config_name="config")
def train(cfg):
    print(cfg.project.model)
    print(cfg.project.dataset)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
    ])
    
    ## Dataset ##
    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.project.data_dir, 
        train=True, 
        transform=transform,
        download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root=cfg.project.data_dir, 
        train=False, 
        transform=transform,
        download=True)

    ## DataLoader ##
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    for batch in train_dataloader:
        inputs, labels = batch
        print(inputs.shape)
        print(labels.shape)
        break

    model = torchvision.models.resnet18(pretrained=True, progress=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
    print(model)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    for batch in train_dataloader:
        inputs, labels = batch
        
        outputs = model(inputs)

        print(outputs.shape)

        loss = criterion(outputs, labels)

        print(loss)
        break


if __name__ == "__main__":
    train()