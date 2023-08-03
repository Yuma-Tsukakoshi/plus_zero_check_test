'''
GPUの都合上, colaboで実装したコードを以下に貼り付けています。
'''

from tqdm.notebook import tqdm
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

# モデルの構築
n_input = 32*32*3
n_output = 10
n_hidden = 128


class CNN(nn.Module):
    def __init__(self, n_output, n_hidden):
        super().__init__()
        # フィルター数(in)、チャネル数(out)、フィルターの正方形
        # チャネル数は次の畳み込みのフィルター数になる
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)  # バッチ正規化レイヤーを追加
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)  # バッチ正規化レイヤーを追加
        self.maxpool = nn.MaxPool2d((2, 2))
        self.l1 = nn.Linear(1152, n_hidden)
        self.dropout1 = nn.Dropout(0.2)  # ドロップアウトレイヤーを追加
        self.l2 = nn.Linear(n_hidden, n_output)
        self.dropout2 = nn.Dropout(0.2)  # ドロップアウトレイヤーを追加
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # バッチ正規化レイヤーで正規化
        x2 = self.maxpool(self.relu(self.bn2(self.conv2(x1))))  # バッチ正規化レイヤーで正規化
        x3 = torch.flatten(x2, 1)
        x4 = self.dropout1(self.relu(self.l1(x3)))  # ドロップアウトレイヤーでドロップアウト
        x5 = self.dropout2(self.l2(x4))  # ドロップアウトレイヤーでドロップアウト
        return x5

#モデルをインスタンス化し、GPUに引き渡している
net = CNN(n_output,n_hidden).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

lr = 0.6
#重みの更新にはモデルのパラメーターが必要
#最適アルゴリズム(確率的勾配降下法)
optimizer = optim.SGD(net.parameters(),lr)

history = np.zeros((0,5))
history

# 学習

num_epoch = 20
F1_list = []
for epoch in range(num_epoch):
    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    n_train, n_test = 0, 0
    f1_train, f1_test = 0, 0

    for inputs, labels in tqdm(train_loader):
        n_train += len(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # 順伝播関数を行っている
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        predicted = torch.max(outputs, 1)[1]

        # テンソルを整数型に変換
        train_loss += loss.item()
        train_acc += (predicted == labels).sum().item()
        f1_train += multiclass_f1_score(input=predicted, target=labels,
                                    num_classes=10, average="macro").item() # F1値を計算
        
    for inputs_test, labels_test in test_loader:
        n_test += len(labels_test)

        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)

        outputs_test = net(inputs_test)

        loss_test = criterion(outputs_test, labels_test)

        predicted_test = torch.max(outputs_test, 1)[1]

        # テンソルを整数型に変換
        val_loss += loss_test.item()
        val_acc += (predicted_test == labels_test).sum().item()
        f1_test += multiclass_f1_score(input=predicted_test, target=labels_test,
                                    num_classes=10, average="macro").item()  # F1値を計算

    train_acc = train_acc/ n_train
    val_acc =  val_acc / n_test
    train_loss = train_loss * batch_size / n_train
    val_loss = val_loss * batch_size /n_test
    F1 = f1_test / n_test
    F1_list.append(F1)

    print(f'Epoch [{epoch+1}/{num_epoch},loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc}]')
    items = np.array([epoch+1,train_loss,train_acc,val_loss,val_acc])
    print(f"F1 score: {F1}")
    history = np.vstack((history,items))

#評価指標 F値
print(F1_list[-1])
