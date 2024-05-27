import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np

class VisionSequentialDataset(Dataset):
    def __init__(self, args, dataset, transform, seqInd, maplab):
        """Create a dataset of context window (history + single future datum)

        Parameters
        ----------
        dataset : _type_
            original torch dataset
        seqInd : _type_
            training sequence indices
        maplab : _type_
            label mapper
        contextlength : int, optional
            length of the history
        """
        self.args = args
        self.dataset = dataset
        self.c = args.conv_proformer["contextlength"]
        self.t = len(seqInd)
        self.time = torch.arange(self.t).float()
        self.seqInd = seqInd
        self.maplab = maplab
        self.transform = transform

        self.max_num_classes = max([len(task) for task in args.task])

    def __len__(self):
        return len(self.seqInd)

    def __getitem__(self, idx):
        # trying
        r = np.random.randint(0, self.t-self.c-1) # select the start of the history
        if self.args.conv_proformer["multihop"]:
            s = np.random.randint(r+self.c, self.t)  # select a 'future' datum
        else:
            s = r+self.c  # select the next datum

        id = list(range(r, r+self.c)) + [s]
        dataid = self.seqInd[id] # get indices for the context window

        data = self.dataset.data[dataid]
        data = self.transform(data)

        time = self.time[id]

        labels = self.dataset.targets[dataid].apply_(self.maplab)
        target = labels[-1].clone() # true label of the future datum

        labels = F.one_hot(labels, self.max_num_classes)
        labels[-1, :] = 0 

        return data, time, labels, target
    
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
        self.contextlength = args.conv_proformer["contextlength"]
        self.train_seqInd = train_seqInd[-self.contextlength:]
        self.test_seqInd = test_seqInd[t:]
        self.maplab = maplab
        self.transform = transform
        
        self.train_time = torch.arange(t).float()
        self.test_time = torch.arange(t, t + len(test_seqInd)).float()

        self.max_num_classes = max([len(task) for task in args.task])
        
    def __len__(self):
        return len(self.test_seqInd)
        
    def __getitem__(self, idx):
        dataid = self.train_seqInd.tolist() + [self.test_seqInd[idx]] # most recent history + inference datum indices

        data = self.dataset.data[dataid]
        data = self.transform(data)

        time = torch.cat([
            self.train_time[-self.contextlength:], 
            self.test_time[idx].view(1)
        ])

        labels = self.dataset.targets[dataid].apply_(self.maplab)
        target = labels[-1].clone() # true label of the future datum

        labels = F.one_hot(labels, self.max_num_classes)
        labels[-1, :] = 0

        return data, time, labels, target
