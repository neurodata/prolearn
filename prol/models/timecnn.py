import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
from .base_trainer import BaseTrainer
import math
from ..utils import get_dataloader

# define the base CNN
class Model(nn.Module):
    """
    Small convolution network with no residual connections (single-head)
    """
    def __init__(self, num_classes=10, channels=3, d_model=128, avg_pool=2, max_len=5000, encoding_type='freq'):
        super(Model, self).__init__()
        self.d_model = d_model
        self.linsize = d_model
        self.max_len = max_len

        # activations
        self.relu = nn.ReLU(inplace=True)

        # conv net
        self.conv1 = nn.Conv2d(channels, d_model, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(d_model)
        self.conv3 = nn.Conv2d(d_model, d_model, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(d_model)
        self.maxpool = nn.MaxPool2d(avg_pool)

        # linear layers
        self.time_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.fc = nn.Linear(self.linsize, num_classes)

        # positional encodings
        if encoding_type == 'vanilla':
            pe = self.get_vanilla_encoding()
        elif encoding_type == 'freq':
            pe = self.get_freq_encoding()
        else:
            raise NotImplementedError
        self.register_buffer('pe', pe)

    def get_vanilla_encoding(self):
        C = 10000
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(C) / (self.d_model)))
        pe = torch.zeros(1, self.max_len, self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_freq_encoding(self):
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = 2 * math.pi / torch.arange(2, self.d_model + 1, 2)
        ffe = torch.zeros(1, self.max_len, self.d_model)
        ffe[0, :, 0::2] = torch.sin(position * div_term)
        ffe[0, :, 1::2] = torch.cos(position * div_term)
        return ffe
    
    def time_encoder(self, t):
        return self.pe[:, t.long(), :].squeeze()

    def forward(self, x, t):
        t = self.time_encoder(t)
        t = self.time_ffn(t)
        t = t[..., None, None]

        x = self.conv1(x)
        x = self.maxpool(self.relu(x))

        x = x + t
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
            "d_model": 128,
            "avg_pool": 2, 
            "encoding_type": 'vanilla'
        }
    elif dataset == 'cifar-10':
        return {
            "channels": 3,
            "avg_pool": 2,
            "d_model": 128,
            "encoding_type": 'vanilla'
        }
    else:
        raise NotImplementedError
    
class Trainer(BaseTrainer):
    def __init__(self, model, dataset, args) -> None:
        super().__init__(model, dataset, args)

    def fit(self, log):
        args = self.args
        nb_batches = len(self.trainloader)
        for epoch in range(args.epochs):
            self.model.train()
            losses = 0.0
            train_acc = 0.0
            for data, time, target in self.trainloader:
                data = data.float().to(self.device)
                time = time.float().to(self.device)
                target = target.long().to(self.device)

                out = self.model(data, time)
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
                log.info(f'{info}')

    def evaluate(self, test_dataset, verbose=False):
        testloader = get_dataloader(
            test_dataset,
            batchsize=100,
            train=False
        )

        self.model.eval()
        with torch.no_grad():
            preds = []
            truths = []
            if verbose:
                progress = tqdm(testloader)
            else:
                progress = testloader
            for data, time, target in progress:
                data = data.float().to(self.device)
                time = time.float().to(self.device)
                target = target.long().to(self.device)

                out = self.model(data, time)

                preds.extend(
                    out.detach().cpu().argmax(1).numpy()
                )
                truths.extend(
                    target.detach().cpu().numpy()
                )
        return np.array(preds), np.array(truths)
    
def main():
    # testing
    x = torch.randn(16, 1, 28, 28)
    t = torch.randint(5000, (16, ))
    net = Model(
        num_classes=2, 
        channels=1, 
        d_model=128,
        encoding_type='vanilla'
    )
    y = net(x, t)
    print(y.shape)

    # x = torch.randn(1, 3, 32, 32)
    # model_kwargs = model_defaults(dataset='cifar-10')
    # net = Model(2, **model_kwargs)
    # y = net(x)
    # print(y.shape)


if __name__ == "__main__":
    main()