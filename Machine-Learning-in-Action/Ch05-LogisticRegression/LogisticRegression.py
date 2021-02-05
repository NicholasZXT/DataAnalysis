
import numpy as np

# cd 'Machine-Learning-in-Action/Ch05-LogisticRegression/'
# cd .\Machine-Learning-in-Action\Ch05-LogisticRegression\


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def gradientAscent(X, y):
    """
    梯度下降法，使用全量数据更新梯度
    @param X: 样本阵
    @param y: 类标签，取值{0,1}
    @return:
    """
    # 数据矩阵需要增加一列全为1，表示截距
    intercep = np.ones((X.shape[0], 1))
    X = np.hstack((intercep, X))
    # 需要将X和y转换为矩阵，以便直接使用矩阵的乘法
    X = np.mat(X)
    y = np.mat(y)
    # 迭代的步长
    alpha = 0.001
    # 最多迭代的次数，这里是写死的
    maxCycles = 500
    # n 是样本数，m 是特征数
    n, m = X.shape
    # 初始化权重
    w = np.ones((m, 1))
    # 开始迭代
    for k in range(maxCycles):
        # 这里的 X 为 n*m, W 是 m*1， X 是样本阵（没有转置）
        # 得到的 h 为 n*1 列向量
        h = sigmoid(X * w)
        error = y - h
        # 下面这个公式就是 LR 的梯度下降法的 权重更新公式的矩阵版
        w = w + alpha * X.transpose() * error
    return w


def gradientAscentCrossEntropy(X, y):
    """
    类标签为 {-1,1}下，使用的是标准的交叉熵损失函数
    :param X:
    :param y: 这里要求 y 使用{1,-1}来表示类标签，而不是{1,0}，
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


def gradientAscentParallel(X, y):
    """
    LR梯度下降法的并行实现
    @param X:
    @param y: 类标签{0,1}
    @return:参数向量 w
    """
    # TODO 这个在本机上不太好实现，涉及的问题有：1.多线程似乎不太合适，需要使用多进程；2. 多进程的话，涉及到进程间通信
    pass


if __name__ == "__main__":
    # 加载数据集
    # % cd Machine-Learning-in-Action\Ch05-LogisticRegression
    dataset = np.loadtxt("testSet.txt")
    X = dataset[:, :2]
    # y切片出来之后是一个一维的array，还需要转成二维的array
    y = dataset[:, -1].reshape((-1, 1))

    # 对比 类标签为{0,1}和{-1,1}下，两种梯度下降方式的结果
    # 可以看出，这两个结果是一样的
    w1 = gradientAscent(X, y)
    w2 = gradientAscentCrossEntropy(X, y)

    # 对比随机梯度下降的结果
    w3 = stochasticGradientAscent(X, y)
    w4 = stochasticGradientAscentImproved(X, y, 200)