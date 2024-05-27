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

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx].unsqueeze(-1)
        label = self.targets[idx]
        return data, label

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
        
    def __len__(self):
        return len(self.test_targets)
        
    def __getitem__(self, idx):
        data = self.test_data[idx].unsqueeze(-1)
        label = self.test_targets[idx]
        return data, label

class VisionSequentialDataset(Dataset):
    def __init__(self, args, dataset, transform, seqInd, maplab):
        """Create the training dataset

        Parameters
        ----------
        dataset : _type_
            original torch dataset
        seqInd : _type_
            training sequence indices
        maplab : _type_
            label mapper
        """
        self.args = args
        self.dataset = dataset
        self.seqInd = seqInd
        self.maplab = maplab
        self.transform = transform

    def __len__(self):
        return len(self.seqInd)

    def __getitem__(self, idx):
        data = self.dataset.data[self.seqInd[idx]]
        data = self.transform(data)
        data = data.flatten(0, -1)
        label = self.dataset.targets[self.seqInd[idx]].apply_(self.maplab)
        return data, label

class VisionSequentialTestDataset(Dataset):
    def __init__(self, args, dataset, transform, train_seqInd, test_seqInd, maplab) -> None:
        """Create the testing dataset

        Parameters
        ----------
        args : _type_
            _description_
        dataset : _type_
            original torch dataset
        train_seqInd : _type_
            training sequence indices
        test_seqInd : _type_
            testing sequence indices
        maplab : _type_
            label mapper
        """
        t = len(train_seqInd)
        self.dataset = dataset
        self.test_seqInd = test_seqInd[t:]
        self.maplab = maplab
        self.transform = transform
        
    def __len__(self):
        return len(self.test_seqInd)
        
    def __getitem__(self, idx):
        data = self.dataset.data[self.test_seqInd[idx]]
        data = self.transform(data)
        data = data.flatten(0, -1)
        label = self.dataset.targets[self.test_seqInd[idx]].apply_(self.maplab)
        return data, label
