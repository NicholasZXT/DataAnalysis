import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

# 用于将tensor压平的匿名函数
flatten = lambda t: torch.flatten(t)

# 上面得到的每张图像是 28x28 的矩阵，下面将每张图片进行了拉直，也就是转换成一个 28x28=784 的向量
mnist_train = datasets.MNIST(root='./datasets/PyTorch/', train=True, download=True,
                             transform=T.Compose([T.ToTensor(),T.Lambda(flatten)]))
mnist_test = datasets.MNIST(root='./datasets/PyTorch/', train=False, download=True,
                            transform=T.Compose([T.ToTensor(),T.Lambda(flatten)]))

# 如果采用CNN网络，就不需要拉平这一步的变换
mnist_train = datasets.MNIST(root='./datasets/PyTorch/', train=True, download=True, transform=T.ToTensor())
mnist_test = datasets.MNIST(root='./datasets/PyTorch/', train=False, download=True, transform=T.ToTensor())

t1 = mnist_train[0][0]
t2 = mnist_train[0][1]

d = mnist_train.data
t3 = mnist_train.__getitem__(0)

# 数据加载器
BATCH_SIZE=10
mnist_train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE)
dataiter = iter(mnist_train_loader)
batch_data, batch_label = next(dataiter)
print(batch_data.__class__)
print(batch_data.shape)

# 卷积层使用
# 2张 4x4x3的图片，3 是通道数，这里将通道数提前了
imgs = np.arange(2*4*4*3, dtype=np.float32).reshape((2,3,4,4))
imgs_tensor = torch.tensor(imgs)

# 卷积层
# 卷积核的形状为 2x2x3x2，其中3对应于in_channels, 2对应于out_channels, 2x2是kernel_size
# 这里步长stride=1，不做padding
conv = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(2, 2), stride=1, padding=0, bias=True, padding_mode='zeros')
res = conv(imgs_tensor)

# 原始图片为 2 张 3通道的 4x4 图片
imgs_tensor.shape
# 经过卷积层之后，变成了 2 张 2通道的 3x3 图片
res.shape


# 卷积函数，可以自定义卷积核
# 自定义的卷积核，包含2个过滤器，分别为 [[1,0], [1,0]], [[0,1],[0,1]]，三个通道都是一样的
kernel = torch.tensor(np.array([1,1,0,0]*3 + [0,0,1,1]*3, dtype=np.float32).reshape((2, 3, 2,2), order='F'))
# 卷积核shape 为 2x3 x 2x2，2对应于输出的 channel——也就是两个过滤器,3对应于输入的channel,2x2是卷积的长宽
kernel.shape

# 使用自定义卷积核的结果
res_2 = nn.functional.conv2d(input=imgs_tensor, weight=kernel, bias=None, stride=1, padding=0)
# 得到的结果为 2 x 2 x3x3, 2张 2通道的 3x3图片
res_2.shape


# ----------池化层------------
# 池化窗口大小为 2x2， 步长=2
pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
# 原始的shape
imgs_tensor.shape
res_3 = pool(imgs_tensor)
res_3.shape

### --------------------RNN练习---------------------
#RNN处理的是序列数据，所以输入数据的维度中，多了一个sequence维度（或者叫time_step），
# 所以输入的数据shape是 (batch_size, seq_length, in_features)
# 以文本处理为例，一般将一个句子看做一个sequence，而句子中的每个单词都用一个词向量来表示，那么一个句子就是一个矩阵 v x s
# v 是 词向量的长度，所有单词都是一样的， s 是这个句子的长度——也就是单词的数量
# 但是文本中，不是所有的句子长度 s 都是一样的，需要做进一步的处理——选择最大的句子长度（或者平均长度）为准，所有的句子都规范化成这个长度
# 超过的部分截断，不足的部分单词补0——这个补充的单词0也对应于一个词向量，这样就得到了形状规范的一个 sequence 样本
# 每个 sequence 对于RNN 来说是一个样本，每次训练使用多个，也就是 batch_size。

# 在设置一层RNN的时候，需要配置的参数有如下几个：in_features——对应于输入层的节点数，hidden_units——隐藏层节点数，也就是状态向量的长度，
# 最重要的是sequence的长度，代表序列的长度.

# 单层、单向 RNN，输入特征数为5，序列长度=4，隐藏层状态向量长度=3，batch_size=2
in_features = 5
seq_length = 4
hidden_size = 3
rnn = nn.RNN(input_size=in_features, hidden_size=hidden_size, num_layers=1, nonlinearity='relu', bias=False,
             batch_first=True, bidirectional=False)

# 输入的数据shape应当为 batch_size * seq_length * in_features
batch_size = 2
rnn_data = torch.randn(batch_size, seq_length, in_features)
rnn_data.shape
res_4 = rnn(rnn_data)
# 得到的结果是一个长度=2 的tuple，第一个元素是输出，第二个元素是隐藏层的状态
# 输出层的shape=(2, 4, 3)，表示 2 个样本，序列长度=4, 输出层也就是隐藏层节点数
res_4[0].shape
res_4[0]
# 隐藏层shape=(1, 2, 3)，1表示只有 1层*单向，2表示批次，3是每个状态向量的长度
# 这个 2 有待研究
res_4[1].shape
res_4[1]


##----------MNIST----------
