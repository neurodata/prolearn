import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from ..utils import get_dataloader

class BaseTrainer:
    def __init__(self, model, dataset, args) -> None:
        self.args = args

        # dataloader
        if args.t > 0:
            self.trainloader = get_dataloader(
                dataset,
                batchsize=args.batchsize,
                train=True
            )

        # model
        self.model = model
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-5
        )

        # learning rate schedule
        if args.t > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=args.epochs * len(self.trainloader)
            )

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # flag
        self.istrained = False

    def fit(self, log):
        args = self.args
        nb_batches = len(self.trainloader)
        for epoch in range(args.epochs):
            self.model.train()
            losses = 0.0
            train_acc = 0.0
            for data, target in self.trainloader:
                data = data.float().to(self.device)
                target = target.long().to(self.device)

                out = self.model(data)
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
        self.istrained = True

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
            for data, target in progress:
                data = data.float().to(self.device)
                target = target.long().to(self.device)

                out = self.model(data)

                preds.extend(
                    out.detach().cpu().argmax(1).numpy()
                )
                truths.extend(
                    target.detach().cpu().numpy()
                )
        return np.array(preds), np.array(truths)