import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
# from ..utils import get_dataloader
# from .base_trainer import BaseTrainer

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, p=0.0):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def prepare_mask_(self, mask):
        attn_mask = torch.zeros_like(mask).float()
        attn_mask.masked_fill_(mask, -torch.inf)
        return attn_mask       
    
    def forward(self, hidden_states, attention_mask, output_attentions=True):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = self.prepare_mask_(attention_mask)
            attention_scores = attention_scores + attention_mask
            # attention_probs *= attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, p=0.0):
        super().__init__()
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim, p),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model, p),
        )
        # self.mha = nn.MultiheadAttention(d_model, num_heads, p, batch_first=True)
        self.mha = MultiHeadAttention(d_model, num_heads, p)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, x, mask):
        # attn_output, weights = self.mha(x, x, x, attn_mask=mask)
        attn_output, weights = self.mha(x, mask)
        out1 = self.layernorm1(x + attn_output)
        ff_output = self.feed_forward(out1)
        out2 = self.layernorm2(out1 + ff_output)
        return out2, weights

class Model(nn.Module):
    def __init__(self, input_size, d_model, num_heads, ff_hidden_dim, num_attn_blocks=1, num_classes=2, max_len=5000, encoding_type='freq'):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.max_len = max_len

        self.attention_blocks = nn.ModuleList(
            [SelfAttention(d_model, num_heads, ff_hidden_dim) for _ in range(num_attn_blocks)]
        )

        self.input_embedding = nn.Linear(input_size+num_classes, d_model//2)
        self.layernorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.classifier = nn.Linear(d_model, num_classes)

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
        div_term = torch.exp(torch.arange(0, self.d_model//2, 2) * (-math.log(C) / (self.d_model//2)))
        pe = torch.zeros(1, self.max_len, self.d_model//2)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_freq_encoding(self):
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = 2 * math.pi / torch.arange(2, self.d_model//2 + 1, 2)
        ffe = torch.zeros(1, self.max_len, self.d_model//2)
        ffe[0, :, 0::2] = torch.sin(position * div_term)
        ffe[0, :, 1::2] = torch.cos(position * div_term)
        return ffe
    
    def time_encoder(self, t):
        enc = torch.cat([self.pe[:, t[i].squeeze().long(), :] for i in range(t.size(0))])
        return enc
        
    def forward(self, data, labels, times, mask):
        u = torch.cat((data, labels), dim=-1)
        u = self.input_embedding(u)

        t = self.time_encoder(times)

        x = torch.cat((u, t), dim=-1)

        for attn_block in self.attention_blocks:
            x, weights = attn_block(x, mask)
        x = torch.select(x, 1, -1)
        x = self.classifier(x)
        return x
    
def model_defaults(dataset):
    if dataset == 'mnist':
        return { 
            "input_size": 28*28,
            "d_model": 256, 
            "num_heads": 8,
            "ff_hidden_dim": 1024,
            "num_attn_blocks": 2,
            "encoding_type": 'freq'
        }
    elif dataset == 'cifar-10':
        return { 
            "input_size": 28*28,
            "d_model": 512, 
            "num_heads": 8,
            "ff_hidden_dim": 2048,
            "num_attn_blocks": 4,
            "encoding_type": 'freq'
        }
    elif dataset == 'synthetic':
        return { 
            "input_size": 1,
            "d_model": 256, 
            "num_heads": 8,
            "ff_hidden_dim": 1024,
            "num_attn_blocks": 2,
            "encoding_type": 'freq'
        }
    else:
        raise NotImplementedError
    
# class Trainer(BaseTrainer):
#     def __init__(self, model, dataset, args) -> None:
#         super().__init__(model, dataset, args)

#     def fit(self, log):
#         args = self.args
#         nb_batches = len(self.trainloader)
#         for epoch in range(args.epochs):
#             self.model.train()
#             losses = 0.0
#             train_acc = 0.0
#             for data, time, label, target in self.trainloader:
#                 data = data.float().to(self.device)
#                 time = time.float().to(self.device)
#                 label = label.float().to(self.device)
#                 target = target.long().to(self.device)

#                 out = self.model(data, label, time)
#                 loss = self.criterion(out, target)

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 losses += loss.item()
#                 self.optimizer.step()
#                 train_acc += (out.argmax(1) == target).detach().cpu().numpy().mean()
#                 self.scheduler.step()
            
#             if args.verbose and (epoch+1) % 10 == 0:
#                 info = {
#                     "epoch" : epoch + 1,
#                     "loss" : np.round(losses/nb_batches, 4),
#                     "train_acc" : np.round(train_acc/nb_batches, 4)
#                 }
#                 log.info(f'{info}')

#     def evaluate(self, test_dataset, verbose=False):
#         testloader = get_dataloader(
#             test_dataset,
#             batchsize=100,
#             train=False
#         )
#         self.model.eval()
#         with torch.no_grad():
#             preds = []
#             truths = []
#             if verbose:
#                 progress = tqdm(testloader)
#             else:
#                 progress = testloader
#             for data, time, label, target in progress:
#                 data = data.float().to(self.device)
#                 time = time.float().to(self.device)
#                 label = label.float().to(self.device)
#                 target = target.long().to(self.device)

#                 out = self.model(data, label, time)

#                 preds.extend(
#                     out.detach().cpu().argmax(1).numpy()
#                 )
#                 truths.extend(
#                     target.detach().cpu().numpy()
#                 )
#         return np.array(preds), np.array(truths)

if __name__ == "__main__":
    # testing
    kwargs = model_defaults('mnist')
    net = Model(num_classes=4, **kwargs)
    data = torch.randn((1, 10, 28*28))
    labels = torch.randn((1, 10, 4))
    times = torch.arange(10).view(1, -1)
    
    mask = torch.zeros(10, 10)
    mask[:5, :5] = 1
    mask = mask.float()

    y = net(data, labels, times, mask)
    print(y.shape)