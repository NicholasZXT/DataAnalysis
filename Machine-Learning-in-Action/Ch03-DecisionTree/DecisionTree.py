from math import log
import numpy as np
import operator
import matplotlib.pyplot as plt


def createDataSet():
    """
    这个函数只是用于生成一个测试数据
    返回值：
    dataSet:一个nested list，每一个sublist包括两部分，前面都是对应于feature的取值，1-yes,0-no,最后一个表示class label
    featureNames：list,表示的是sublist中各个feature的名称，因为dataSet中没有存储feature的名称，只存储了值
    """
    # data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    data = np.array([[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]])
    label = np.array(['yes', 'yes', 'no', 'no', 'no'])
    featureNames = ['no surfacing', 'flippers']
    return data, label, featureNames


def shannonEntropy(dataLabel):
    """
    这个函数用于计算当前数据集data的香农熵，而不是直接计算使得information gain最大的feature
    计算数据集的香农熵不需要特征，只需要这个数据集对应的类标签就行了。
    :param dataLabel: np.array对应data每一行观察的类标签，注意，这个类标签是字符，而不是数字
    :return: shannonEnt，返回的是当前数据集的香农熵
    """
    # 首先计算entry的数量
    numEntries = len(dataLabel)
    # 统计dataLabel中类的个数,以及每一类的数量，用dict存储
    classLabels = {}
    for label in dataLabel:
        classLabels[label] = classLabels.get(label, 0) + 1
    # 开始计算信息熵
    shannonEnt = 0
    for classCounts in classLabels.values():
        # prob是每一类的比例，这个比例作为概率分布看待
        prob = float(classCounts) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitData(data, label, feature, value):
    """
    用于根据给定的feature以及value划分数据集，返回的是data中feature=value的数据，
    并且 feature 所在的列会被删除
    :param data: np.array格式
    :param label: 类标签
    :param feature: 用于划分的特征在data中的索引，int类型
    :param value: 用于划分data的feature所取的值
    :return: subData，返回的子集，subLabel，对应的类标签
    """
    subData = data[data[:, feature] == value]
    subData = np.delete(subData, feature, 1)
    subLabel = label[data[:, feature] == value]
    return subData, subLabel


def chooseBestFeature(data, label):
    """
    从data中选择最佳分割的feature
    :param data: np.array，
    :param label: 类标签
    :return: bestFeature，最佳分割的feature在data中的索引
    """
    numFeature = data.shape[1]
    # 当前数据集本身的香农熵，其实这个不用计算，因为它对于每个特征都是一样的
    baseEntropy = shannonEntropy(dataLabel=label)
    # 初始化最佳信息增益和最佳feature的索引
    bestInfoGain = 0.0
    bestFeature = -1
    # 循环计算每个特征的分割后的信息增益
    for feature in range(numFeature):
        # 获得该feature所取值的个数
        uniqueValues = set(data[:, feature])
        # 初始化分割后数据集的香农熵
        newEntropy = 0
        # 对于当前feature的每个值，进行分割数据，同时计算每个子集的香农熵，最后得到分割后的香农熵之和
        for value in uniqueValues:
            subData, subLabel = splitData(data, label, feature, value)
            prob = subData.shape[0]/float(data.shape[0])
            newEntropy += prob * shannonEntropy(subLabel)
        # 计算该feature的信息增益
        infoGain = baseEntropy - newEntropy
        # 更新最大的信息增益和对应的feature
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = feature
    return bestFeature


def majorVote(data, label):
    """
    这个函数用于处理特征使用完之后，叶子节点中仍然含有多类的数据，此时该叶子节点的类标签由多数投票决定
    :param data: np.array类型，其实不需要使用data，只需要使用label就行了
    :param label:
    :return: keyMax，返回计数最多的类标签
    """
    labelCount = {}
    for classLabel in label:
        labelCount[classLabel] = labelCount.get(classLabel, 0) + 1
    labelMax = max(labelCount, key=labelCount.get)
    return labelMax


def trainDecisionTree(data, label, featureNames):
    """
    ID3算法，递归创建一棵决策树，创建的决策树是以nested dict的形式存在的；
    这个nested dict的每一层key,是feature和feature的value交替进行，value不是类标签，就是子树(sub-dict)。
    示例如：{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    :param data: np.array类型
    :param label: 类标签
    :param featureNames: data中每一列特征的名称
    :return: tree，嵌套的字典，字典的key是特征名称
    """
    # 判断递归是否停止有两个情况：
    # 递归结束的第一个条件，数据集的所有类标签都属于同一类
    if len(set(label)) == 1:
        return label[0]
    # 递归结束的第二个条件，用完了所有的特征，但是数据集还是含有多个类别，这时使用上面的majorVote给出类标签
    if len(data) == 0:
        return majorVote(data, label)
    # 上述两个都不满足，说明还可以再进行分割
    # 首先找出当前dataSet下的最佳分割feature(index表示)
    bestFeature = chooseBestFeature(data, label)
    bestFeatureName = featureNames[bestFeature]
    # 删除已使用过的特征名称
    del featureNames[bestFeature]
    # 根据当前数据集构造树，使用这个最佳feature构建一棵树
    # 先用这个feature作为子树的根节点
    tree = {bestFeatureName: {}}
    # 拿到最佳特征的所有取值
    featureValues = set(data[:, bestFeature])
    for value in featureValues:
        # 对于最佳特征的每个取值进行遍历，获得相应的数据子集
        subData, subLabel = splitData(data, label, bestFeature, value)
        # 对于每个数据子集，构造相应的决策树，注意，这里使用的featureNames已经去掉了使用过的最佳feature的名称
        tree[bestFeatureName][value] = trainDecisionTree(subData, subLabel, featureNames[:])
    return tree


def predictDecisionTree(tree, featureNames, testData):
    """
    决策树分类函数
    :param tree: 已经由trainDecisionTree创建好的决策树，nested dict
    :param featureNames: 特征的名称
    :param testData: 待分类的数据，np.array类型
    :return: classLabel，类标签
    示例如：{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    """
    # 首先，对于testDate，需要知道决策树tree中用于分割的第一个feature（根节点）是哪个，
    # 这个信息就是tree这个nested dict中的key，虽然这个key只有一个，还是需要使用index=0
    rootFeatureName = list(tree.keys())[0]
    # 得到根节点名称在featureNames中对应的index
    rootFeatureIndex = featureNames.index(rootFeatureName)
    # 得到根节点下的子树，这些子树的每个key就是根节点特征所取的不同值
    subTree = tree[rootFeatureName]
    # 判断testData中对应于根节点特征的值符合子树中的哪个分支
    for value in subTree.keys():
        if testData[rootFeatureIndex] == value:
            # 找到这个分支后，看这个分支的value是不是还是一个dict,是的话，说明还有子树，还需要进一步判断，
            if type(subTree[value]).__name__ == 'dict':
                # classLabel = predictDecisionTree(subTree, featureNames, testData),注意这里传入subTree是不对的
                classLabel = predictDecisionTree(subTree[value], featureNames, testData)
            # 如果不是dict，说明就到了叶子节点，直接返回叶子节点的类标签
            else:
                classLabel = subTree[value]
    return classLabel


# -----测试上面的代码
if __name__ == "__main__":

    data, label, featureNames = createDataSet()

    tree = trainDecisionTree(data, label, featureNames.copy())

    testData = data[0, :]
    predictDecisionTree(tree, featureNames, testData)

    predLabel = []
    for i in range(len(data)):
        testData = data[i, :]
        t = predictDecisionTree(tree, featureNames, testData)
        predLabel.append(t)

    label
    predLabel
