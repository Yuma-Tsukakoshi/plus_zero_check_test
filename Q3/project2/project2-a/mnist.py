import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#GPUの利用確認
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)



