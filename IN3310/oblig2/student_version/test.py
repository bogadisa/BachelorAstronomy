import torch
from torch import nn


features = torch.rand(128, 10, 512)

maxpool = nn.MaxPool1d(1)
print(maxpool(torch.transpose(features, 1, 2)).squeeze().shape)