"""
Implementation of Wide-Resnet
Adapted from an open-source implementation.
https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
"""
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from ..utils import get_dataloader


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class Model(nn.Module):
    def __init__(self, 
            num_classes=10, 
            depth=10, 
            base_chans=4, 
            input_chans=3, 
            widen_factor=1, 
            dropRate=0.0, 
            max_len=5000,
            d_model=128,
            encoding_type='freq'
        ):
        super(Model, self).__init__()
        nChannels = [
            base_chans, 
            base_chans*widen_factor, 
            2*base_chans*widen_factor, 
            4*base_chans*widen_factor
        ]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(input_chans, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.nChannels = nChannels[3]

        self.max_len = max_len
        self.d_model = d_model
        self.time_ffn = nn.Sequential(
            nn.Linear(d_model, nChannels[1]),
            nn.ReLU()
        )

        # positional encodings
        if encoding_type == 'vanilla':
            pe = self.get_vanilla_encoding()
        elif encoding_type == 'freq':
            pe = self.get_freq_encoding()
        else:
            raise NotImplementedError
        self.register_buffer('pe', pe)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

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

        out = self.conv1(x)
        out = self.block1(out)
        out = out + t
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # out = F.avg_pool2d(out, 8)
        out = self.pool(out) # for different input sizes
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
def model_defaults(dataset):
    if dataset == 'cifar-10':
        return {
            "depth": 16,
            "base_chans": 16, 
            "input_chans": 3,
            "widen_factor": 4,
            "dropRate": 0,
            "d_model": 128,
            "encoding_type": 'freq'
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
    x = torch.randn(16, 3, 32, 32)
    t = t = torch.randint(5000, (16, ))
    model_kwargs = model_defaults("cifar-10")
    net = Model(
        num_classes=2,
        **model_kwargs
    )
    y = net(x, t)
    print(y.shape)


if __name__ == "__main__":
    main()