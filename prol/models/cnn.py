import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
from .base_trainer import BaseTrainer

class Model(nn.Module):
    """
    Small convolution network with no residual connections (single-head)
    """
    def __init__(self, num_classes=10, channels=3, avg_pool=2, lin_size=320):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(channels, 80, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(avg_pool)

        self.linsize = lin_size
        self.fc = nn.Linear(self.linsize, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))

        x = self.conv2(x)
        x = self.maxpool(self.relu(self.bn2(x)))

        x = self.conv3(x)
        x = self.maxpool(self.relu(self.bn3(x)))
        x = x.flatten(1, -1)

        x = self.fc(x)
        return x
    
def model_defaults(dataset):
    if dataset == 'mnist':
        return {
            "channels": 1,
            "avg_pool": 2, 
            "lin_size": 80
        }
    elif dataset == 'cifar-10':
        return {
            "channels": 3,
            "avg_pool": 2, 
            "lin_size": 320
        }
    else:
        raise NotImplementedError
    
class Trainer(BaseTrainer):
    def __init__(self, model, dataset, args) -> None:
        super().__init__(model, dataset, args)
    
def main():
    # testing
    x = torch.randn(1, 1, 28, 28)
    net = Model(10, 1, 2, 80)
    y = net(x)
    print(y.shape)

    x = torch.randn(1, 3, 32, 32)
    model_kwargs = model_defaults(dataset='cifar-10')
    net = Model(2, **model_kwargs)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    main()