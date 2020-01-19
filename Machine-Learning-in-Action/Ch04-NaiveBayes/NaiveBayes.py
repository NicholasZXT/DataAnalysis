
# %cd Machine-Learning-in-Action/Ch04-NaiveBayes
# cd .\Machine-Learning-in-Action\Ch04-NaiveBayes\

import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB


def trainDataSet():
    """
    这个函数创建了一个用于训练的数据集.
    它含有5个句子，每个句子是一个list，类标签0/1表示是否含有abusive的语句
    注意，每个句子不是等长的
    :return:
    """
    docs=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 is abusive, 0 not
    label = [0, 1, 0, 1, 0, 1]
    return docs, label


def testDataSet():
    """
    这个函数用于创建测试数据集
    :return: 返回的是一个list，包含两个句子，同时也返回了实际的类别
    """
    docs=[['love', 'my', 'dalmation'], ['stupid', 'garbage']]
    labels=np.array([0, 1])
    return docs, labels


def createVocabulary(dataset):
    """
    这个函数用于将dataset中所有documents里使用过的word提取出来，组成一个set——也就是word 的 dictionary
    :param dataset: nested list，每一个sub-list是一个句子(不一定等长)
    :return: vocabList:返回值，是一个list
    """
    # 首先创建一个空的set
    vocabSet=set([])
    for document in dataset:
        # | 表示取 set 的并
        vocabSet = vocabSet | set(document)
    # 这里要将set转换成list,后面才方便使用(因为要使用index，而set是不能使用index的)
    vocabList = list(vocabSet)
    return vocabList


def setOfWords2Vec(vocabList,inputDoc):
    """
    这个函数根据vocabList这个word的词典，将inputDoc这个输入转化成词向量
    使用的是 set-of-word representation
    :param vocabList: 用于匹配的word的字典,list类型
    :param inputDoc: list，它是已经经过分词处理过的document
    :return: docVetcor:一个长度和vocabList等长的0-1向量，np.array类型
    """
    # 首先创建一个和vocabList等长的初始化vector，全为0
    # list * n,表示把这个list重复n遍
    docVector = [0] * len(vocabList)
    # 开始统计
    for word in inputDoc:
        if word in vocabList:
            # vocabList.index(word)返回的是word这个内容的index
            # 这里不管word出现了几次，vocabList中该位置都是直接设为 1
            docVector[vocabList.index(word)] = 1
        else:
            print("the word %s is not present in the vocabList"%word)
    return np.array(docVector)


def bagOfWords2Vec(vocabList, inputDoc):
    """
    这个函数也是根据vocabList这个词典，将inputDoc这个输入转化成词向量
    但使用的是 bag-of-word representation
    :param vocabList: 用于匹配的word的字典,list类型
    :param inputDoc: list，它是已经经过分词处理过的document
    :return: docVetcor:也一个长度和vocabList等长的向量，但是其中每个位置记录的是该word出现的次数
    """
    docVetcor = [0]*len(vocabList)
    for word in inputDoc:
        if word in vocabList:
            # 这一句不一样了，不再是直接赋值1，而是累加计数
            docVetcor[vocabList.index(word)] += 1
        else:
            print("the word %s is not present in the vocabList"%word)
    return np.array(docVetcor)


# -------- 朴素贝叶斯的训练算法----------------

def trainBinaryBernoulliNB(dataArray, classLabels):
    """
    二分类 Bernoulli NB 的训练函数,这个函数特别适用于处理文档数据，
    每一个文档都依据vocabList(长度为M)处理成了等长的0-1向量，这些训练数据可以
    组成一个NxM的(稀疏)matrix，N为文档数量，M为词向量的长度，也就是NB中feature的个数，
    因为是二分类问题，类标签也只有0-1两种
    :param dataArray: 是一个np.array类型，N x M，注意，不是np.matrix
    :param classLabels: np.array类型，是一个0-1向量
    :return:
    pyPositive:一个数，表示P(Y=1)的概率
    py1ConditionVec:一个向量，array类型，是P(Xi=1|Y=1)的概率向量，因为Xi也是0-1，所以P(Xi=0|Y=1)=1-P(Xi=1|Y=1)
    py0ConditionVec:也是一个向量，array类型，是P(Xi=1|Y=0)的概率向量
    """
    # 首先计算训练实例个数，array的行数
    numOfDocs = dataArray.shape[0]
    # 计算每个训练实例的feature个数，
    numOfFeatures = dataArray.shape[1]
    # 计算正类的先验概率P(Y=1)
    pyPositive=sum(classLabels)/float(numOfDocs)
    # Laplace smoothing, lamda=1，但是大多数文献里 类的先验概率似乎不需要进行平滑
    # pyPositive=(sum(classLabels)+1)/float(numOfDocs+2)

    # 分布初始化各个特征的正类条件概率py1ConditionVec和负类条件概率py0ConditionVec
    # py1ConditionVec=np.zeros(numOfFeatures)
    # py0ConditionVec=np.zeros(numOfFeatures)
    # Laplace smoothing 下的初始化，lamda = 1
    py1ConditionVec = np.ones(numOfFeatures)
    py0ConditionVec = np.ones(numOfFeatures)

    # 统计Y=1时各个Xi=1出现的次数，对于Y=0也是这样
    for i in range(numOfDocs):
        if classLabels[i] == 1:
            py1ConditionVec += dataArray[i]
        else:
            py0ConditionVec += dataArray[i]

    # 计算P(X= 1 |Y=1)和P(X= 1 |Y=0)这两个概率向量，这里使用了np.array的element-wise性质
    # P(X= 0 |Y=1)和 P(X= 0 |Y=0)只要用1减去就可以得到了
    # py1Vector=py1Vector/float(sum(classLabels))
    # py0Vector=py0Vector/float(numOfDocs-sum(classLabels))
    # Laplace smoothing, lambda=1
    py1ConditionVec = py1ConditionVec/float(sum(classLabels)+2)
    py0ConditionVec = py0ConditionVec/float(numOfDocs-sum(classLabels)+2)
    return pyPositive, py1ConditionVec, py0ConditionVec


def classifyBinaryBernoulliNB(docArray, pyPositive, py1ConditionVec, py0ConditionVec):
    """
    二分类 Bernoulli NB 的分类函数
    使用的是 set-of-bag 模型表示文档向量
    输入：
    docArray:待分类的文档向量(0-1)矩阵，N x M,已经经过set-of-bag model下的vocabulary处理
    pyPositive,py1ConditionVec,py0ConditionVec都是trainNaiveBayes返回的概率参数，np.array类型
    注意，这些概率都是未经过对数变换的
    输出：
    prediction:np.array,N x 3，这三列分别是：predClass,log(py1),log(py0)
    """
    numOfDocs = len(docArray)
    prediction = np.zeros((numOfDocs, 3))
    # 对于每个文档进行分类
    for i in range(numOfDocs):
        py1 = np.log(pyPositive)
        py0 = np.log(1-pyPositive)
        docVector = docArray[i]
        # 下面这个for 循环是根据原本的求和公式计算的
        # numOfFeatures=len(py1ConditionVec)
        # for i in range(numOfFeatures):
        #     py1+=docVector[i]*np.log(py1ConditionVec[i])+(1.0-docVector[i])*np.log((1-py1ConditionVec[i]))
        # 使用np.array的element-wise方式计算更为简便
        # py1 += sum(docVector*np.log(py1ConditionVec)+(1-docVector)*np.log(1-py1ConditionVec))
        # py0 += sum(docVector*np.log(py0ConditionVec)+(1-docVector)*np.log(1-py0ConditionVec))
        py1 += sum(docVector*np.log(py1ConditionVec))
        py0 += sum((1-docVector)*np.log(py0ConditionVec))
        if py1 >= py0:
            predClass = 1
            prediction[i] = np.array([predClass, py1, py0])
        else:
            predClass = 0
            prediction[i] = np.array([predClass, py1, py0])
    return prediction


def trainMultinomialNB(dataArray, classLabels):
    """
    二分类 Multinomial NB 模型的训练函数
    注意，返回的pyPositive没有经过对数变换，只有条件概率做了对数变换
    :param dataArray: np.array类型, N x M, N个训练数据，
    注意这里的dataArray每个位置表示的是对应vocabulary中单词出现的次数，这个表示方式使得后续的计算比较容易
    :param classLabels: np.array类型, N x 1
    :return:
    pyPositive:一个数，表示P(Y=1)的类先验概率, 没有经过对数变换
    py1ConditionVec:一个向量，array类型，是P(Xi|Y=1)的概率向量
    py0ConditionVec:也是一个向量，array类型，是P(Xi|Y=0)的概率向量，没有经过对数变换
    """
    # 首先计算训练实例个数，也就是行数
    numOfDocs=dataArray.shape[0]
    # 计算每个训练实例的向量长度，也就是vocabulary的长度，但是这里不能称为feature
    numOfVocabulary=dataArray.shape[1]
    # 计算正类的先验概率P(Y=1)
    pyPositive=sum(classLabels)/float(numOfDocs)

    # 计算条件概率向量
    # 初始化向量
    # 各个word的计算方式和之前是一样的，
    # py1ConditionVec=np.zeros(numOfVocabulary)
    # py0ConditionVec=np.zeros(numOfVocabulary)
    # 但是 Multinomial NB 条件概率的分母是该类下所有文档的单词长度之和，这和Bernoulli不一样
    # py1WordsNum=0
    # py0WordsNum=0
    # Laplace Smoothing的初始化
    py1ConditionVec = np.ones(numOfVocabulary)
    py0ConditionVec = np.ones(numOfVocabulary)
    py1WordsNum = numOfVocabulary
    py0WordsNum = numOfVocabulary
    for i in range(numOfDocs):
        if classLabels[i] == 1:
            py1ConditionVec += dataArray[i]
            # dataArray[i]是一个整数向量，分量之和就是这个文档所有含的单词数
            py1WordsNum += sum(dataArray[i])
        else:
            py0ConditionVec += dataArray[i]
            py0WordsNum += sum(dataArray[i])
    py1ConditionVec = py1ConditionVec/float(py1WordsNum)
    py0ConditionVec = py0ConditionVec/float(py0WordsNum)
    return pyPositive, py1ConditionVec, py0ConditionVec


def classifyMultinomialNB(docArray, pyPositive, py1ConditionVec, py0ConditionVec):
    """
    二分类 Multinomial NB 的分类函数
    :param docArray: 待分类的文档向量(正整数向量)矩阵,N x M，已经经过 bag-of-words model 处理
    :param pyPositive:
    :param py1ConditionVec:
    :param py0ConditionVec:
    上面三个参数都是trainBinaryMultinomialNB返回的概率参数，np.array类型，这些概率没有经过对数变换
    :return: prediction:np.array,N x 3，这三列分别是：predClass,log(py1),log(py0)
    """
    numOfDocs=docArray.shape[0]
    # 初始化预测结果的数组
    prediction=np.zeros((numOfDocs, 3))
    # 对每个文档分别进行分类
    for i in range(numOfDocs):
        py1 = np.log(pyPositive)
        py0 = np.log(1-pyPositive)
        # 取第i个文档的特征
        docVector = docArray[i]
        py1 += sum(docVector*np.log(py1ConditionVec))
        py0 += sum(docVector*np.log(py0ConditionVec))
        if py1 >= py0:
            predClass = 1
            prediction[i] = np.array([predClass, py1, py0])
        else:
            predClass = 0
            prediction[i] = np.array([predClass, py1, py0])
    return prediction


# ---------处理文本数据，转换成可以处理的数字格式

dataList, trainLabel = trainDataSet()
vocabList = createVocabulary(dataList)
v1 = setOfWords2Vec(vocabList, dataList[0])
v2 = bagOfWords2Vec(vocabList, dataList[0])
trainSet = np.zeros((len(dataList), len(vocabList)))
trainBag = np.zeros((len(dataList), len(vocabList)))
for i in range(len(dataList)):
    trainSet[i] = setOfWords2Vec(vocabList, dataList[i])
    trainBag[i] = bagOfWords2Vec(vocabList, dataList[i])

testList, testLabel = testDataSet()
testSet = np.zeros((len(testList), len(vocabList)))
testBag = np.zeros((len(testList), len(vocabList)))
for i in range(len(testList)):
    testSet[i] = setOfWords2Vec(vocabList, testList[i])
    testBag[i] = bagOfWords2Vec(vocabList, testList[i])


pyPositive, py1Cond, py0Cond = trainBinaryBernoulliNB(trainSet, trainLabel)
predLabel = classifyBinaryBernoulliNB(testSet, pyPositive, py1Cond, py0Cond)

pyPositive, py1Cond, py0Cond = trainMultinomialNB(trainSet, trainLabel)
predLabel = classifyMultinomialNB(testSet, pyPositive, py1Cond, py0Cond)

# ---------使用sklearn的库进行对比
#  对比后可以发现，得到结果是一样的，nice！！！

berNB = BernoulliNB()
berNB.fit(trainSet, trainLabel)
berNB.class_log_prior_
t = berNB.feature_log_prob_

mulNB = MultinomialNB()
mulNB.fit(trainBag, trainLabel)
mulNB.class_log_prior_
t = mulNB.feature_log_prob_

np.log(pyPositive)
np.log(py1Cond)
np.log(py0Cond)

