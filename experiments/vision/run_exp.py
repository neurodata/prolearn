'''
Boilerplate code for the exps
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

from prol.models.proformer import TransformerClassifier
from prol.models.cnn import SmallConv

class SetParams:
    def __init__(self, dict) -> None:
        for k, v in dict.items():
            setattr(self, k, v)

def get_cycle(N: int) -> list:
    """Get the primary cycle

    Parameters
    ----------
    N : int
        time between two task switches

    Returns
    -------
    list
        primary cycle
    """
    return [1] * N + [0] * N

def get_torch_dataset():
    """Get the original torch datase

    Returns
    -------
    _type_
        torch dataset
    """
    dataset = torchvision.datasets.MNIST(
            root="data",
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.Lambda(lambda x : torch.flatten(x))
            ]),
            download=True
        )
    # normalize
    tmp = dataset.data.float() / 255.0
    tmp = (tmp - 0.5)/0.5
    dataset.data = tmp
    return dataset

def get_task_indicies_and_map(tasks: list, y: np.ndarray):
    """Get the indices for each task + the label mapping

    Parameters
    ----------
    tasks : list
        task specification e.g. [[0, 1], [2, 3]]
    y : np.ndarray
        dataset targets
    """
    tasklib = {}
    for i, task in enumerate(tasks):
        tasklib[i] = []
        for lab in task:
            tasklib[i].extend(
                np.where(y == lab)[0].tolist()
            )
    mapdict = {}
    for task in tasks:
        for i, lab in enumerate(task):
            mapdict[lab] = i
    maplab = lambda lab : mapdict[lab]
    return tasklib, maplab

def get_sequence_indices(N, total_time_steps, tasklib, seed=1996):
    """Get indices for a sequence drawn from the stochastic process

    Parameters
    ----------
    N : time between two task switches
    total_time_steps : length of the sequence drawn
    tasklib : original task indices
    seed : random seed

    Returns
    -------
    index sequence
    """
    unit = get_cycle(N)
    pattern = np.array((unit * math.ceil(total_time_steps/(len(unit))))[:total_time_steps]).astype("bool")
    seqInd = np.zeros((total_time_steps,)).astype('int')
    np.random.seed(seed)
    seqInd[pattern] = np.random.choice(tasklib[0], sum(pattern), replace=False)
    seqInd[~pattern] = np.random.choice(tasklib[1], sum(~pattern), replace=False)
    return seqInd

class SequentialDataset(Dataset):
    def __init__(self, dataset, seqInd, maplab, contextlength=200):
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
        self.dataset = dataset
        self.contextlength = contextlength
        self.t = len(seqInd)
        self.time = torch.arange(self.t).float()
        self.seqInd = seqInd
        self.maplab = maplab

    def __len__(self):
        return len(self.seqInd)

    def __getitem__(self, idx):
        r = np.random.randint(0, len(self.seqInd)-2*self.contextlength) # select the start of the history
        s = np.random.randint(r+self.contextlength, r+2*self.contextlength)  # select a 'future' datum

        id = list(range(r, r+self.contextlength)) + [s]
        dataid = self.seqInd[id] # get indices for the context window

        data = self.dataset.data[dataid]
        data = data.view(data.shape[0], data.shape[-1]**2)
        labels = self.dataset.targets[dataid].apply_(self.maplab)
        time = self.time[id]
        
        target = labels[-1].clone() # true label of the future datum
        labels[-1] = np.random.binomial(1, 0.5) # replace the true label of the future datum with a random label

        return data, time, labels, target
    
class SequentialTestDataset(Dataset):
    def __init__(self, dataset, train_seqInd, test_seqInd, maplab, contextlength) -> None:
        t = len(train_seqInd)
        self.dataset = dataset
        self.contextlength = contextlength
        self.train_seqInd = train_seqInd[-contextlength:]
        self.test_seqInd = test_seqInd[t:]
        self.maplab = maplab
        
        self.train_time = torch.arange(t).float()
        self.test_time = torch.arange(t, t + len(test_seqInd)).float()
        
    def __len__(self):
        return len(self.test_seqInd)
        
    def __getitem__(self, idx):
        dataid = self.train_seqInd.tolist() + [self.test_seqInd[idx]] # most recent history + inference datum indices

        data = self.dataset.data[dataid]
        data = data.view(data.shape[0], data.shape[-1]**2)
        labels = self.dataset.targets[dataid].apply_(self.maplab)
        time = torch.cat([
            self.train_time[-self.contextlength:], 
            self.test_time[idx].view(1)
        ])

        target = labels[-1].clone() # true label of the future datum
        labels[-1] = np.random.binomial(1, 0.5) # replace the true label of the future datum with a random label

        return data, time, labels, target

class Trainer:
    def __init__(self, model, dataset, args) -> None:
        self.args = args

        self.trainloader = DataLoader(dataset, batch_size=args.batchsize)

        self.model = model
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

    def run(self):
        args = self.args
        nb_batches = len(self.trainloader)
        for epoch in range(args.epochs):
            self.model.train()
            losses = 0.0
            train_acc = 0.0
            for data, time, label, target in self.trainloader:
                data = data.float().to(self.device)
                time = time.float().to(self.device)
                label = label.float().to(self.device)
                target = target.long().to(self.device)

                out = self.model(data, label, time)
                loss = self.criterion(out, target)

                self.optimizer.zero_grad()
                loss.backward()
                losses += loss.item()
                self.optimizer.step()
                train_acc += (out.argmax(1) == target).detach().cpu().numpy().mean()
                self.scheduler.step()
            
            if args.verbose and (epoch+1) % 10 == 0:
                info = {
                    "epoch" : epoch + 1,
                    "loss" : np.round(losses/nb_batches, 4),
                    "train_acc" : np.round(train_acc/nb_batches, 4)
                }
                print(info)

    def evaluate(self, testloader):
        self.model.eval()
        preds = []
        truths = []
        for data, time, label, target in tqdm(testloader):
            data = data.float().to(self.device)
            time = time.float().to(self.device)
            label = label.float().to(self.device)
            target = target.long().to(self.device)

            out = self.model(data, label, time)

            preds.extend(
                out.detach().cpu().argmax(1).numpy()
            )
            truths.extend(
                target.detach().cpu().numpy()
            )
        return np.array(preds), np.array(truths)
        
def plotting(y, ci, args):
    t = args.t
    T = args.T
    N = args.N
    time = np.arange(t, T)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, y, c="k", lw=2)
    ax.fill_between(time, y-ci, y+ci, alpha=0.2, color='k')

    unit = get_cycle(N)
    pattern = np.array((unit * math.ceil((T)/(2*N))))[t:T].astype("bool")

    for i in time[pattern]:
        ax.add_artist(Rectangle((i, 0), 1, 1, alpha=0.4, edgecolor=None, facecolor="blue"))
    for i in time[~pattern]:
        ax.add_artist(Rectangle((i, 0), 1, 1, alpha=0.4, edgecolor=None, facecolor="orange"))

    ax.set_xlabel("time")
    ax.set_ylabel("Instantaneous Risk")
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([time[0], time[-1]])

    plt.show()
    plt.savefig("test.png", bbox_inches="tight")

def main():
    # input parameters
    args = SetParams({
        "N": 20,                    # time between two task switches                   
        "t": 2000,                  # training time
        "T": 2000,                  # future time horizon
        "task": [[0, 1], [2, 3]],   # task specification
        "contextlength": 200,       
        "seed": 1996,              
        "image_size": 28,           
        "device": "cuda:3",             
        "lr": 1e-3,         
        "batchsize": 128,
        "epochs": 150,
        "verbose": True,
        "reps": 100                 # number of test reps
    })

    # get source dataset
    torch_dataset = get_torch_dataset()
    
    # get indices for each task
    taskInd, maplab = get_task_indicies_and_map(
        tasks=args.task,
        y=torch_dataset.targets.numpy()
    )

    # get a training sequence
    train_SeqInd = get_sequence_indices(
        N=args.N, 
        total_time_steps=args.t, 
        tasklib=taskInd, 
        seed=args.seed
    )

    # sample a bunch of test sequences
    test_seqInds = [
        get_sequence_indices(args.N, args.T, taskInd, seed=args.seed+1000*(rep+1))
        for rep in range(args.reps)
    ]

    # form the train dataset
    train_dataset = SequentialDataset(
        dataset=torch_dataset, 
        seqInd=train_SeqInd,
        maplab=maplab,
        contextlength=args.contextlength
    )

    # model
    model = TransformerClassifier(
        input_size=args.image_size ** 2,
        d_model=512, 
        num_heads=8,
        ff_hidden_dim=2048,
        num_attn_blocks=4,
        num_classes=2, 
        contextlength=200
    )
    
    # train
    trainer = Trainer(model, train_dataset, args)
    trainer.run()

    # evaluate
    preds = []
    truths = []
    for i in range(args.reps):
        # form a test dataset for each test sequence
        test_dataset = SequentialTestDataset(
            torch_dataset, 
            train_SeqInd,
            test_seqInds[i],
            maplab,
            args.contextlength
        )
        testloader = DataLoader(
            test_dataset, 
            batch_size=100,
            shuffle=False
        )
        preds_rep, truths_rep = trainer.evaluate(testloader)
        preds.append(preds_rep)
        truths.append(truths_rep)
    preds = np.array(preds)
    truths = np.array(truths)

    # compute metrics
    mean_error = np.mean(preds != truths, axis=0).squeeze()
    std_error = np.std(preds != truths, axis=0).squeeze()
    ci = std_error * 1.96/np.sqrt(args.reps).squeeze()

    err = np.mean(preds != truths)
    print(f"error = {err:.4f}")

    # plot
    plotting(mean_error, ci, args)


if __name__ == "__main__":
    main()