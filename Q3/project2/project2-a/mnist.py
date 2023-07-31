'''
colaboで実装したコードを以下に貼り付けています。
'''
#ライブラリのインポート
from tqdm.notebook import tqdm
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

data_root = './data'

# 前処理
# 1. データをテンソルに変換 ToTensor
# 2. 正規化する　Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    # view: tensorを別のtensorに変換 つじつま合わせ 平滑化
    transforms.Lambda(lambda x: x.view(-1)),
])

# dataの呼び出し

# train True : 訓練用のデータセット = 6万件
train_set = datasets.MNIST(
    root=data_root,
    train=True,
    download=True,
    transform=transform
)

# test False : 検証用集合を取得  = 1万件
test_set = datasets.MNIST(
    root=data_root,
    train=False,
    download=True,
    transform=transform
)

# dataの確認
image, label = train_set[0]
print(len(train_set))
print('Image tyoe', type(image))
print('Image shape', image.shape)

image, label = test_set[0]
print(len(test_set))
print('Image tyoe', type(image))
# print('Image shape',image.shape)

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
n_input = 784
n_output = 10
n_hidden1 = 256
n_hidden2 = 128


class Net(nn.Module):
  def __init__(self, n_input, n_output, n_hidden1, n_hidden2):
    super().__init__()
    self.l1 = nn.Linear(n_input, n_hidden1)
    self.l2 = nn.Linear(n_hidden1, n_hidden2)
    self.l3 = nn.Linear(n_hidden2, n_output)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x1 = self.l1(x)
    x2 = self.sigmoid(x1)
    x3 = self.l2(x2)
    x4 = self.relu(x3)
    x5 = self.l3(x4)

    return x5

#モデルをインスタンス化し、GPUに引き渡している
net = Net(n_input,n_output,n_hidden1,n_hidden2).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

lr = 0.5
#重みの更新にはモデルのパラメーターが必要
#最適アルゴリズム(確率的勾配降下法)
optimizer = optim.SGD(net.parameters(),lr)

history = np.zeros((0, 5))
history

# 学習

num_epoch = 20
for epoch in range(num_epoch):
  train_acc, train_loss = 0, 0
  val_acc, val_loss = 0, 0
  n_train, n_test = 0, 0

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

  train_acc = train_acc / n_train
  val_acc = val_acc / n_test
  train_loss = train_loss * batch_size / n_train
  val_loss = val_loss * batch_size / n_test

  print(f'Epoch [{epoch+1}/{num_epoch},loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc}]')
  items = np.array([epoch+1, train_loss, train_acc, val_loss, val_acc])
  history = np.vstack((history, items))

print(f'初期状態: 損失:  {history[0,3]:.5f} 精度: {history[0,4]:.5f}')
print(f'最終状態: 損失:  {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}')

plt.rcParams['figure.figsize'] = (8, 6)
plt.plot(history[:, 0], history[:, 1], 'b', label='train')
plt.plot(history[:, 0], history[:, 3], 'k', label='test')
plt.xlabel('iter')
plt.ylabel('loss')
plt.title('loss carve')
plt.legend()
plt.show()

plt.rcParams['figure.figsize'] = (8, 6)
plt.plot(history[:, 0], history[:, 2], 'b', label='train')
plt.plot(history[:, 0], history[:, 4], 'k', label='test')
plt.xlabel('iter')
plt.ylabel('acc')
plt.title('accuracy')
plt.legend()
plt.show()
