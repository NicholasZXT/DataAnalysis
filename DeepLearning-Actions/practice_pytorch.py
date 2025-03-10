# packages import
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
import spacy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms as T

LOCAL_DATA_PATH = r"D:\Project-Workspace\Python-Projects\DataAnalysis\local-datasets"
BATCH_SIZE = 10

# %% ---------------- CNN练习 -------------------------
def __CNN():
    # 用于将tensor压平的匿名函数
    flatten = lambda t: torch.flatten(t)
    data_path = os.path.join(LOCAL_DATA_PATH, r'PyTorch\MNIST')
    # 上面得到的每张图像是 28x28 的矩阵，下面将每张图片进行了拉直，也就是转换成一个 28x28=784 的向量
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True,
                                 transform=T.Compose([T.ToTensor(), T.Lambda(flatten)]))
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True,
                                transform=T.Compose([T.ToTensor(), T.Lambda(flatten)]))
    # 如果采用CNN网络，就不需要拉平这一步的变换
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True, transform=T.ToTensor())
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True, transform=T.ToTensor())

    # t1 = mnist_train[0][0]
    # t2 = mnist_train[0][1]
    # d = mnist_train.data
    # t3 = mnist_train.__getitem__(0)

    # 数据加载器
    mnist_train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE)
    dataiter = iter(mnist_train_loader)
    batch_data, batch_label = next(dataiter)
    print(batch_data.__class__, ', ', batch_data.shape)

    # 卷积层使用
    # 2张 4x4x3的图片，3 是通道数，这里将通道数提前了
    imgs = np.arange(2*4*4*3, dtype=np.float32).reshape((2, 3, 4, 4))
    imgs_tensor = torch.tensor(imgs)

    # 卷积层
    # 卷积核的形状为 2x2x3x2，其中3对应于in_channels, 2对应于out_channels, 2x2是kernel_size
    # 这里步长stride=1，不做padding
    conv = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(2, 2), stride=1, padding=0, bias=True, padding_mode='zeros')
    res = conv(imgs_tensor)

    # 原始图片为 2 张 3通道的 4x4 图片
    print(imgs_tensor.shape)
    # 经过卷积层之后，变成了 2 张 2通道的 3x3 图片
    print(res.shape)

    # 卷积函数，可以自定义卷积核
    # 自定义的卷积核，包含2个过滤器，分别为 [[1,0], [1,0]], [[0,1],[0,1]]，三个通道都是一样的
    kernel = torch.tensor(np.array([1, 1, 0, 0] * 3 + [0, 0, 1, 1] * 3, dtype=np.float32).reshape((2, 3, 2, 2), order='F'))
    # 卷积核shape 为 2x3 x 2x2，2对应于输出的 channel——也就是两个过滤器,3对应于输入的channel,2x2是卷积的长宽
    print(kernel.shape)

    # 使用自定义卷积核的结果
    res_2 = nn.functional.conv2d(input=imgs_tensor, weight=kernel, bias=None, stride=1, padding=0)
    # 得到的结果为 2 x 2 x3x3, 2张 2通道的 3x3图片
    print(res_2.shape)

    # --------------- 池化层 ----------------
    # 池化窗口大小为 2x2， 步长=2
    pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
    # 原始的shape
    print(imgs_tensor.shape)
    res_3 = pool(imgs_tensor)
    print(res_3.shape)


# %% -------------------- RNN练习拟合sin函数 ---------------------
def __RNN_sin():
    pass

# RNN处理的是序列数据，所以输入数据的维度中，多了一个sequence维度（或者叫time_step），
# 所以输入的数据shape是 (batch_size, seq_length, in_features)
# 以文本处理为例，一般将一个句子看做一个sequence，而句子中的每个单词都用一个词向量来表示，那么一个句子就是一个矩阵 v x s
# v 是 词向量的长度，所有单词都是一样的， s 是这个句子的长度——也就是单词的数量
# 但是文本中，不是所有的句子长度 s 都是一样的，需要做进一步的处理——选择最大的句子长度（或者平均长度）为准，所有的句子都规范化成这个长度
# 超过的部分截断，不足的部分单词补0——这个补充的单词0也对应于一个词向量，这样就得到了形状规范的一个 sequence 样本
# 每个 sequence 对于RNN 来说是一个样本，每次训练使用多个，也就是 batch_size。

# 在设置一层RNN的时候，需要配置的参数有如下几个：in_features——对应于输入层的节点数，hidden_units——隐藏层节点数，也就是状态向量的长度，
# 最重要的是sequence的长度，代表序列的长度.
# 需要注意的是，RNN输入数据的shape=(batch_size, time_steps, input_size)，特别要注意 time_steps 这个维度.

# 构造数据集
TRAINING_EXAMPLES, TESTING_EXAMPLES = 3000, 500
SAMPLE_INTERVAL, TIME_STEPS = 0.01*np.pi, 10

def generate_data(seq):
    """
    从序列seq中进行采样，用 [i: i+TIME_STEPS-1] 作为输入值X， 第 i+TIMES_STEPS 项作为Y值.
    即用sin函数前面的 TIME_STEPS 个点的信息，预测第 i+TIME_STEPS 个点的函数值.
    样本数为 序列长度 - TIME_STEPS
    """
    X, Y = [], []
    # 样本数为：序列长度 - TIME_STEPS
    for i in range(len(seq)-TIME_STEPS):
        # 序列的 [i: TIME_STEPS-1 ] 子序列作为输入；第 i+TIME_STEPS 项作为输出
        X.append(seq[i: (i+TIME_STEPS)])
        Y.append(seq[i+TIME_STEPS])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def generate_sin_dataset():
    # 确定测试数据起点，这之前都是训练数据集，训练数据集的取值区间是 [0, test_start]，这中间取 TRAINING_EXAMPLES + TIME_STEPS 个点，
    test_start = (TRAINING_EXAMPLES + TIME_STEPS) * SAMPLE_INTERVAL
    # 确定测试数据集的终点，测试数据集的取值区间是 [test_start, test_end]， 这中间取 TESTING_EXAMPLES + TIME_STEPS 个点
    test_end = test_start + (TESTING_EXAMPLES + TIME_STEPS) * SAMPLE_INTERVAL
    # 从 sin 中产生数据点
    sin_x_train = np.linspace(0, test_start, TRAINING_EXAMPLES + TIME_STEPS, dtype=np.float32)
    sin_y_train = np.sin(sin_x_train)
    sin_x_test = np.linspace(test_start, test_end, TESTING_EXAMPLES + TIME_STEPS, dtype=np.float32)
    sin_y_test = np.sin(sin_x_test)
    # 生成训练数据集
    # 注意，这里是要用sin值序列的 [i: TIME_STEPS-1] 项 预测 第 i+TIMES_STEPS 项，所以只需要 seq_train_y 值，不需要横坐标的值
    train_X, train_y = generate_data(sin_y_train)
    test_X, test_y = generate_data(sin_y_test)
    # 每个训练样本是一个长度(time_step)=10的 sequence，sequence中每个位置的值是一个长度=1的向量（也就是标量）
    train_X = train_X.reshape(TRAINING_EXAMPLES, TIME_STEPS, 1)
    train_y = train_y.reshape(TRAINING_EXAMPLES, 1)
    test_X = test_X.reshape(TESTING_EXAMPLES, TIME_STEPS, 1)
    test_y = test_y.reshape(TESTING_EXAMPLES, 1)
    # 训练样本数太大的话，下面转成tensor会报错
    train_X = torch.tensor(train_X)
    train_y = torch.tensor(train_y)
    test_X = torch.tensor(test_X)
    test_y = torch.tensor(test_y)
    return train_X, train_y, test_X, test_y

train_X, train_y, test_X, test_y = generate_sin_dataset()
print(train_X.shape, ', ', train_y.shape, ', ', test_X.shape, ', ', test_y.shape)

# 测试一下 RNN 的层
# input_size=1，隐藏层是长度=5的向量
# rnn = nn.RNN(input_size=1, hidden_size=5, num_layers=1, nonlinearity='tanh', bias=False, batch_first=True)
# res = rnn(test_X)
# # 输出层
# res[0].shape
# # 隐藏层（状态层）
# res[1].shape


# %%  训练模型
class RnnSinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=5, num_layers=1, nonlinearity='relu', bias=False, batch_first=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=50, out_features=1, bias=True)

    def forward(self, X):
        rnn_out = self.rnn(X)
        # 输出层向量拉平
        rnn_out_flat = self.flatten(rnn_out[0])
        out = self.linear(rnn_out_flat)
        return out

def RNN_train():
    # 初始化模型
    sin_rnn = RnnSinModel()
    # 定义损失函数
    mse = nn.MSELoss()
    # 定义优化器
    optimizer = optim.SGD(params=sin_rnn.parameters(), lr=0.1)

    for epoch in range(1, 31):
        # 获取预测值
        sin_pred = sin_rnn(train_X)
        # sin_pred.shape
        # 清空梯度
        optimizer.zero_grad()
        # 计算损失函数
        loss = mse(sin_pred, train_y)
        # 计算梯度
        loss.backward()
        # 使用优化器更新参数
        optimizer.step()
        # 新的MSE损失
        sin_pred_new = sin_rnn(train_X)
        train_loss = mse(sin_pred_new, train_y)
        pred_y = sin_rnn(test_X)
        test_loss = mse(pred_y, test_y)
        # print(loss)
        print("epoch {:2} ---- training MSE-Loss  is：{:.4f}".format(epoch, train_loss))
        print("epoch {:2} ---- testing  MSE-Loss  is：{:.4f}".format(epoch, test_loss))


# %% ------------------IMDB数据集加载---------------------
def __IMDB():
    pass


class ImdbDataset(Dataset):
    def __init__(self, filepath, part=None):
        super(ImdbDataset, self).__init__()
        self.filepath = filepath
        self.__nlp = spacy.load("en_core_web_sm")
        self.doc_vector, self.label = self.transform(filepath, part)

    def transform(self, filepath, part):
        text_data = load_files(filepath)
        text_list, label = text_data.data, text_data.target
        if part is not None:
            text_list = text_list[:part]
            label = label[:part]
        text = [doc.decode("utf-8").replace("<br />", " ") for doc in text_list]
        sentence_vec = [self.sentence2vector(self.__nlp(doc)) for doc in text]
        # 上面得到的 sentence_vec 中，每个句子的长度都不一样，需要选择最短的句子做截断处理-----这个处理的不好
        # seq_length = min([sentence.shape[0] for sentence in sentence_vec])
        # sentence_vec_trunc = [sentence[:seq_length, :] for sentence in sentence_vec]
        # doc_len = [sentence.shape for sentence in sentence_vec_trunc]
        # print("seq_length", seq_length)
        # print("doc_len", doc_len)
        # doc_vector = np.array(sentence_vec_truc)

        # 选择句子的平均长度，不足的进行padding
        seq_length = int(np.mean(([sentence.shape[0] for sentence in sentence_vec])))
        sentence_vec_pad = [self.sentence_vector_padding(sentence, seq_length) for sentence in sentence_vec]
        doc_vector = np.array(sentence_vec_pad)
        return doc_vector, label

    def sentence2vector(self, doc):
        """
        用于从 sentence 中过滤出有用的word
        @param sentence: spacy的doc对象，封装了一个句子
        @return: 句子中有效单词所组成的 词向量矩阵,
        """
        doc_matrix = doc.tensor
        # 接下来要从上面剔除掉 标点符号 等无效部分的词向量
        index_list = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_currency:
                continue
            else:
                index_list.append(token.i)
        result = doc_matrix[index_list, :]
        # print(result.shape)
        return result

    def sentence_vector_padding(self, sen_vec, seq_length):
        """
        对每个句子对应的词向量矩阵进行填充，保证句子长度一样
        """
        if sen_vec.shape[0] >= seq_length:
            return sen_vec[:seq_length, :]
        else:
            pad_len = int(seq_length-sen_vec.shape[0])
            sen_vec_pad = np.pad(array=sen_vec, pad_width=((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            return sen_vec_pad

    def __len__(self):
        return self.doc_vector.shape[0]

    def __getitem__(self, item):
        return self.doc_vector[item, :, :], self.label[item]


imdb_train = ImdbDataset(filepath=os.path.join(LOCAL_DATA_PATH, r'aclImdb/train'), part=20)
imdb_test = ImdbDataset(filepath=os.path.join(LOCAL_DATA_PATH, r'aclImdb/test'), part=10)
# doc_vector = imdb_train.doc_vector
# doc_vector.shape
# doc_label = imdb_train.label
# doc_label.shape

# 构建数据加载器
imdb_train_loader = DataLoader(dataset=imdb_train, batch_size=BATCH_SIZE)
imdb_test_loader = DataLoader(dataset=imdb_test, batch_size=BATCH_SIZE)
# imdb_train_loader = DataLoader(dataset=imdb_train, batch_size=batch_size, num_workers=4)
# imdb_test_loader = DataLoader(dataset=imdb_test, batch_size=batch_size, num_workers=4)
# # 检查使用
# iter_batch = iter(imdb_train_loader)
# X, y = next(iter_batch)
# print(X.__class__,  y.__class__)
# print(X.shape, y.shape)


# %% ------------- 使用LSTM来进行IMDB的文本分类 ------------------
class ImdbLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_layers):
        super().__init__()
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=seq_length*hidden_size, out_features=2, bias=True)

    def forward(self, X):
        lstm_out = self.lstm_layer(X)
        lstm_flatten = self.flatten(lstm_out[0])
        y_pred = self.linear(lstm_flatten)
        return y_pred


def Imbd_train():
    # 初始化模型
    lstm_rnn = ImdbLSTM(input_size=96, hidden_size=48, seq_length=20, num_layers=1)
    # 定义损失函数
    crossEnt = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.SGD(params=lstm_rnn.parameters(), lr=0.1)
    # 测试
    # out = lstm_rnn(X)
    # loss = crossEnt(out, y.long())
    # 开始迭代训练
    for epoch in range(1, 5):
        train_loss = 1000.0
        for batch_id, (X, y) in enumerate(imdb_train_loader):
            y_pred = lstm_rnn(X)
            optimizer.zero_grad()
            loss = crossEnt(y_pred, y.long())
            loss.backward()
            optimizer.step()

            y_pred_new = lstm_rnn(X)
            train_loss = crossEnt(y_pred_new, y.long())
        print("train_loss: {:.4f}".format(train_loss))