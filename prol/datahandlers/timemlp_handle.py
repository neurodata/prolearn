import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticSequentialDataset(Dataset):
    def __init__(self, args, x, y):
        """Create the training dataset

        Parameters
        ----------
        args : _type_
            configs
        x : torch tensor
            inputs
        y : torch tensor
            labels
        """
        self.data = x
        self.targets = y
        self.t = len(x)
        self.time = torch.arange(self.t).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx].unsqueeze(-1)
        time = self.time[idx]
        label = self.targets[idx]
        return data, time, label

class SyntheticSequentialTestDataset(Dataset):
    def __init__(self, args, x_train, y_train, x, y) -> None:
        """Create the testing dataset

        Parameters
        ----------
        args : _type_
            configs
        x_train : torch tensor
            train inputs
        y_train : torch tensor
            train labels
        x : torch tensor
            test inputs
        y : torch tensor
            test labels
        """
        t = len(y_train)
        self.test_data = x[t:]
        self.test_targets = y[t:]

        self.test_time = torch.arange(t, t + len(self.test_data)).float()
        
    def __len__(self):
        return len(self.test_targets)
        
    def __getitem__(self, idx):
        data = self.test_data[idx].unsqueeze(-1)
        time = self.test_time[idx]
        label = self.test_targets[idx]
        return data, time, label