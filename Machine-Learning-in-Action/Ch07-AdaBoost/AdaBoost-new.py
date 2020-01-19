

import numpy as np
import pandas as pd


def stumpClassify(data, featureIndex, thresh, inequation):
    """
    这个函数是一个决策树桩的分类器，对于data，根据featureIndex和给定的thres来进行分类，
    inequation是不等式的方向，它用于决定大于thresh标记为-1还是小于thresh标记为-1
    :param data:
    :param featureIndex:
    :param thresh:
    :param inequation:
    :return:
    """
    result = np.ones((data.shape[0], 1))
    if inequation == "lessThan":
        result[data[:, featureIndex] <= thresh] = -1.0
    else:
        result[data[:, featureIndex] > thresh] = -1.0
    return result


def buildStump(data, y, weight):
    """
    这个函数用于在数据集data中构建一颗误差最小的决策树桩
    这里的误差最小指的是在权重weight下的最小
    :param data:
    :param y:
    :param weight:
    :return:
    """
    n, m = data.shape
    # 步长数
    numSteps = 10.0
    # 初始化一个空的决策树桩
    bestStump = {}
    # 初始化决策树桩的预测值，之所以要保留这个预测值是因为adaboost里要使用
    bestStumpPred = np.zeros((m, 1))
    # 初始化最优误差
    minError = np.inf

    # 在所有的特征中迭代
    for featureIndex in range(0, m):
        rangeMin = data[:, featureIndex].min()
        rangeMax = data[:, featureIndex].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        # 在选定特征的取值列表中迭代，
        # 从-1开始 不知道是处于什么考虑。。。。
        for step in range(-1, int(numSteps)+1):
            # 在不等式的两个方向中迭代
            # 这是因为只知道决策树桩的特征和thresh还不够，还需要知道不等式的方向来判定分为1还是-1
            for inequation in ["lessThan", "greaterThan"]:
                thresh = rangeMin + step*stepSize
                # 根据特征和thresh来确定该决策树桩的预测值
                predictedValues = stumpClassify(data, featureIndex, thresh, inequation)
                error = np.ones((n, 1))
                error[predictedValues == y] = 0
                # 计算加权的误差
                weightedError = np.sum(error * weight)
                if weightedError < minError:
                    print('''split featureIndex:{feature}, thresh:{value}, 
                    thresh inequation:{inequ}, weighted error: {error}
                    '''.format(feature=featureIndex, value=thresh, inequ=inequation, error=minError))
                    minError = weightedError
                    bestStump["featureIndex"] = featureIndex
                    bestStump['thresh'] = thresh
                    bestStump['inequation'] = inequation
                    # 还需要这个最佳决策树桩的预测值
                    bestStumpPred = predictedValues.copy()
    return bestStump, minError, bestStumpPred


def adaBoostTrain(data, y, numIter):
    """

    :param data:
    :param y:
    :param numIter:
    :return:
    """
    n = data.shape[0]
    # 初始化空的基本分类器列表
    weakClassfiers = []
    # 初始化权重
    weight = np.ones((n,1)) / n
    # 这个记录的是累积的加权预测值之和，也就是最终的加法分类器结果
    # 这里初始化为0
    aggStumpPred = np.zeros((n, 1))
    # 迭代numIter次
    for i in range(numIter):
        # 根据当前的权重weight计算数据data的最优决策树桩
        stump, error, stumpPred = buildStump(data, y, weight)
        # 计算当前分类器的权重alpha
        # alpha = float(0.5 * np.log((1.0 - error) / error))
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        stump['alpha'] = alpha
        # 将当前的决策树桩加入到弱分类器列表中
        weakClassfiers.append(stump)
        # 更新数据集的权重
        weight = weight*(np.exp( -1*alpha*y*stumpPred))
        weight = weight/weight.sum()
        # 计算累积的预测误差，这个预测误差是已有分类器累加后的分类结果
        aggStumpPred += alpha*stumpPred
        # 计算已有累加分类器的分类误差
        aggError = (np.sign(aggStumpPred) != y) * np.ones((n, 1))
        errorRate = aggError.sum()/n
        if errorRate == 0.0 :
            break
    return weakClassfiers


def adaBoostPred(testData, adaClassifier):
    """

    :param testData:
    :param adaClassifier:
    :return:
    """
    aggClassPred = 0
    for i in range(len(adaClassifier)):
        classPred = stumpClassify(testData,
                                  adaClassifier[i]['featureIndex'],
                                  adaClassifier[i]['thresh'],
                                  adaClassifier[i]['inequation'],)
        aggClassPred += adaClassifier[i]['alpha']*classPred
    return np.sign(aggClassPred)


# -----测试上述算法

if __name__ == "__main__":

    data = np.array([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    y = np.array([1.0, 1.0, -1.0, -1.0, 1.0]).reshape(-1,1)

    adaclassifier = adaBoostTrain(data, y, numIter=40)
    # [{'featureIndex': 0,
    #   'thresh': 1.3,
    #   'inequal': 'lessThan',
    #   'alpha': 0.6931471805599453},
    #  {'featureIndex': 1,
    #   'thresh': 1.0,
    #   'inequal': 'lessThan',
    #   'alpha': 0.9729550745276565},
    #  {'featureIndex': 0,
    #   'thresh': 0.9,
    #   'inequal': 'lessThan',
    #   'alpha': 0.8958797346140273}]

    testData = data[[0], :]
    adaBoostPred(testData, adaclassifier)