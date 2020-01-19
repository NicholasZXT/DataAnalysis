
import numpy as np

# cd 'Machine-Learning-in-Action/Ch05-LogisticRegression/'
# cd .\Machine-Learning-in-Action\Ch05-LogisticRegression\

dataSet = np.loadtxt("testSet.txt")
X = dataSet[:, :2]
# y切片出来之后是一个一维的array，还需要转成二维的array
y = dataSet[:, -1].reshape((-1, 1))


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def gradientAscent(X, y):
    # 数据矩阵需要增加一列全为1，表示截距
    intercep = np.ones((X.shape[0],1))
    X = np.hstack((intercep,X))
    # 需要将X和y转换为矩阵，以便直接使用矩阵的乘法
    X = np.mat(X)
    y = np.mat(y)
    # 迭代的步长
    alpha = 0.001
    # 最多迭代的次数，这里是写死的
    maxCycles = 500
    n, m = X.shape
    # 初始化权重
    w = np.ones((m, 1))
    # 开始迭代
    for k in range(maxCycles):
        h = sigmoid(X * w)
        error = y - h
        w = w + alpha * X.transpose() * error
    return w


def gradientAscentCrossEntropy(X, y):
    """
    这个是用交叉熵损失函数表示
    :param X:
    :param y: 这里要求 y 使用{1,-1}来表示类标签，而不是{1,0}
    :return:
    """
    # 数据矩阵X需要增加一列全为1，表示截距
    intercep = np.ones((X.shape[0], 1))
    X = np.hstack((intercep, X))
    # 将 y 里的 0 换成-1
    y = np.where(y > 0, y, -1)
    # 需要将x和y转换为矩阵，以便直接使用矩阵的乘法
    X = np.mat(X)
    # 需要注意的是，这里y还要转成对角矩阵用于向量化迭代
    # y本身是二维的,shape为(n,1)，需要压平成一维array，否则diag函数就是提取对角线的元素，而不是生成对角矩阵
    y = np.diag(y.flatten())
    y = np.mat(y)
    # 迭代的步长
    alpha = 0.001
    # 最多迭代的次数，这里是写死的
    maxCycles = 500
    n, m = X.shape
    # 初始化权重
    w = np.ones((m, 1))
    for i in range(maxCycles):
        h = sigmoid(-y*X*w)
        w = w + alpha*X.transpose()*y*h
    return w


# 可以看出，这两个结果是一样的
w1 = gradientAscent(X, y)
# w2 = gradientAscentCrossEntropy(X, y)


def stochasticGradientAscent(X, y):
    """
    这里随机梯度的迭代次数是训练集的样本个数，循环选择样本，不是随机选择样本
    :param X:
    :param y:
    :return:
    """
    # 数据矩阵X需要增加一列全为1，表示截距
    intercep = np.ones((X.shape[0],1))
    X = np.hstack((intercep,X))
    # 需要将X和y转换为矩阵，以便直接使用矩阵的乘法
    X = np.mat(X)
    y = np.mat(y)
    # 迭代的步长
    alpha = 0.001
    # 最多迭代的次数，这里是写死的
    maxCycles = 500
    n, m = X.shape
    # 初始化权重
    w = np.ones((m, 1))
    # 开始迭代
    # 这里是在训练集上迭代，运行完其实只是迭代了一次
    for i in range(n):
        i = 1
        h = sigmoid(X[i] * w)
        error = y[i] - h
        w = w + alpha * X[i].transpose() * error
    return w


def stochasticGradientAscentImproved(X, y, numIter=150):
    # 数据矩阵X需要增加一列全为1，表示截距
    intercep = np.ones((X.shape[0],1))
    X = np.hstack((intercep,X))
    # 需要将X和y转换为矩阵，以便直接使用矩阵的乘法
    X = np.mat(X)
    y = np.mat(y)
    # 最多迭代的次数，这里是写死的
    maxCycles = 500
    n, m = X.shape
    # 初始化权重
    w = np.ones((m, 1))
    # 开始迭代
    # 这里的迭代是在训练集上迭代
    for j in range(numIter):
        # 使用索引来表示X中对应的样本，然后通过随机抽取索引的方式来实现随机抽取样本
        dataIndex = list(range(n))
        for i in range(n):
            # 迭代步长动态变化
            alpha = 4/(1.0 + i + j) + 0.01
            # 随机选择一个样本
            randIndex = np.random.choice(len(dataIndex))
            h = sigmoid(X[randIndex] * w)
            error = y[randIndex] - h
            w = w + alpha * X[randIndex].transpose() * error
            # 抽取之后就要删除，也就是不放回抽取
            del dataIndex[randIndex]
    return w


w3 = stochasticGradientAscent(X, y)
w4 = stochasticGradientAscentImproved(X, y, 200)