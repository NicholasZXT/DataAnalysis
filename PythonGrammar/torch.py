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