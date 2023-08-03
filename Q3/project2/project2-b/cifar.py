import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

#GPUの利用確認
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_root = './data'

# 前処理
# 1. データをテンソルに変換 ToTensor
# 2. 正規化するNormalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

# dataの呼び出し
train_set = datasets.CIFAR10(
    root=data_root,
    train=True,
    download=True,
    transform=transform
)

test_set = datasets.CIFAR10(
    root=data_root,
    train=False,
    download=True,
    transform=transform
)

# ミニバッチ処置

batch_size = 500
# data_set , batch_size , shuffle

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True
)
