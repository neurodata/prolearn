import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
from .base_trainer import BaseTrainer
from ..utils import get_dataloader
import math

class Model(nn.Module):
    """
    multi-layer perceptron
    """
    def __init__(self, num_classes=10, input_size=784, hidden_size=512, d_model=128, max_len=5000):
        super(Model, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = self.get_freq_encoding()
        self.register_buffer('pe', pe)

        self.net = nn.Sequential(
            nn.Linear(input_size + d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

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
        u = torch.cat((x, t), dim=-1)
        return self.net(u)
    
def model_defaults(dataset):
    if dataset == 'mnist':
        return {
            "input_size": 28*28,
            "hidden_size": 512
        }
    elif dataset == 'cifar-10':
        return {
            "input_size": 32*32*3,
            "hidden_size": 512
        }
    elif dataset == 'synthetic':
        return {
            "input_size": 1,
            "hidden_size": 256
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
    x = torch.randn(16, 1)
    t = torch.randint(5000, size=(16,))
    net = Model(2, 1, 512)
    y = net(x, t)
    print(y.shape)


if __name__ == "__main__":
    main()