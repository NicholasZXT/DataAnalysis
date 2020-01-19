'''
This is my own pratice for KNN algorithm

part I:
group,labels=createDataSet()
knn(target=[0,0],dataset=group,labels=labels,k=3)

part II:
trainData,trainLabels=file2matrix('datingTestSet.txt')
labels=labels2int(trainLabels)
plt.scatter(trainData[:,1],trainData[:,2],c=labels)
loocvRate=0.1
knnLoocv(trainData,trainLabels,k=3,loocvRate=loocvRate)

part III:
trainData,trainLabels=dir2array('trainingDigits')
testingData,testingLabels=dir2array('testDigits')
handwritingClassify(trainData,trainLabels,testingData,testingLabels,k=3)

'''
import numpy as np
import operator
import os
from importlib import reload


def createDataSet():
    '''
    这个函数只是为了简短的测试后面的 knn 函数提供数据而已。
    '''
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


def knn(target, dataset, labels, k):
    '''
    target 是待分类目标,格式是 np.array；
    dataset 是训练数据，格式是 np.array，每一行为一个观测;
    labels 为对应类标签,这里假定类标签是字符串组成的 list；
    k 是 KNN 的参数。
    '''
    #step 1:首先计算target到 dataset 中的各点的距离
    #观测个数
    N=dataset.shape[0]
    #特征个数  
    num_of_features=dataset.shape[1]
    #得到target 与 dataset 中的坐标值之差,使用tile时是将target重复N次，然后利用 np的 element-wise 运算做减法
    #得到和dataset一样的array  
    diff=np.tile(target,(N,1))-dataset
    #计算欧式距离
    distance_square=diff**2
    distance=(distance_square.sum(axis=1))**0.5

    #step-2:
    #将距离进行排序，由小至大
    distance_sort=distance.argsort()
    #选择距离最小的 k 个(前 k 个)
    index=distance_sort[0:k]

    #找出前 k 个中数量占优的类标签，因为 labels 是字符串，并且这里不局限于二分类，所以不是那么容易处理。
    #以下在循环过程中一边访问，一边统计各个类标签的个数，
    #注意，这里是以 dict的形式存储的。
    #首先要创建一个空dict，然后逐步加入类标签和相应的count
    labels_count={}
    # labels_k存储的distance排序后前k个的labels,这种使用列表生成式来一次访问list多个元素的方式很有用
    #不过实际上，后面的语句中并没有使用这个 labels_k,而是直接使用的 index
#    labels_k=[labels[i] for i in index]  
    for i in index:
        #label_vote 是每次循环中得到的类标签，是字符串格式
        label_vote=labels[i]
        #下面这个很有技巧性
        labels_count[label_vote]=labels_count.get(label_vote,0)+1
    #以下使用operator中的itemgetter()和sorted()函数对dict进行排序
    #注意，这里一定要使用 .items() 来取出dict中的内容，返回值是一个list
    labels_count_sort=sorted(labels_count.items(),key=operator.itemgetter(1),reverse=True)
    #因为sorted返回的是一个list，类似于 [('john', 15), ('jane', 12), ('dave', 10)],我们要的是label，也就是第一个元素的第一项
    target_label=labels_count_sort[0][0]
    return target_label


def file2matrix(filename):
    '''
    读取名为 filename 的文本文件，然后以 np.array 格式返回其中的数值数据。
    Mat 是数据阵，每一行为一个观测，格式为 np.array；
    ClassLabels 是字符变量组成的 list。

    这个函数只是为了2.2节中的数据输入而设立的，和 KNN 本身没什么关系。
    这里要读取的数据文件是 datingTestSet.txt，它总共有4列，前三列是 variable，最后一列是 classlabel。
    '''
    with open(filename) as f:
        #获取文件中数据的行数——也就是观测个数
        NumberOfLines=len(list(f))
    Mat=np.zeros((NumberOfLines,3))
    ClassLabels=[]
    index=0
    with open(filename) as f:
        for line in f.readlines():
            line=line.strip()
            line=line.split('\t')
            Mat[index,:]=np.array(line[0:3],dtype=np.float)
            ClassLabels.append(line[-1])
            index+=1
    return Mat,ClassLabels

def labels2int(labels):
    '''
    这是我自己写的函数，用于将上一个 file2matrix 中得到 list 格式的 Classlabels 转换为整数，
    方便 scatter 中作图。
    返回的 labesl 已经是一个 np.array
    '''
    #这个 class_map 中存放的是字符串形式的类标签和int 的对应形式，格式为 dictionary
    class_map={};
    for i in labels:
        class_map[i]=class_map.get(i,0)+1
    #实际上得到的 class_map 中类标签对应的整数就是该类标签出现的次数。
    newLabels=[class_map[i] for i in labels]
    return np.array(newLabels)


def autoNorm(data):
    '''
    这个函数用于将 file2matrix 中得到的data 的各列分别标准化，以消除量纲的影响。
    data为 np.array 格式，返回值 normData 也是 array。
    由于测试数据也同样需要被标准化，所以这里的 min 和max 也需要被返回，或者返回 min 和 range 也可以。
    这里返回的是最小值和范围
    '''
    num_of_features=data.shape[1]
    N=data.shape[0]
    dataMax=data.max(axis=0)
    dataMin=data.min(axis=0)
    dataRange=dataMax-dataMin
    minValues=dataMin
    rangeValues=dataRange
    dataMax=np.tile(dataMax,(N,1))
    dataMin=np.tile(dataMin,(N,1))
    dataRange=np.tile(dataRange,(N,1))
    newData=(data-dataMin)/dataRange
    return newData,minValues,rangeValues



def knnLoocv(data,labels,k,loocvRate):
    '''
    这个函数是上面一些函数的封装，也是用 KNN 分类，但是使用了 类似LOOCV的方式。
    loocvRate是用作 test 的数据集比例，这里简单的选择了前面的一部分作为了 testset。
    返回值是
    '''
    # loocvRate=0.1
    # data,labels=file2matrix("datingTestSet.txt")
    normData,minValues,rangeValues=autoNorm(data)
    # labelsInt=labels2int(labels)
    N=normData.shape[0]
    numTest=int(N*loocvRate)
    testData=normData[:numTest]
    # testLabels=labelsInt[:numTest]
    testLabels=labels[0:numTest]
    trainData=normData[numTest:N]
    # trainLabels=labelsInt[numTest:N]
    trainLabels=labels[numTest:N]
    errorCount=0
    # k=3
    for i in range(0,numTest):
        classlabel=knn(testData[i],trainData,trainLabels,k)
        if (classlabel!=testLabels[i]):
            errorCount+=1
    errorRate=errorCount/float(numTest)
    print("the total error rate was ",errorRate)




#以下是用于2.3节的代码，用KNN分类手写数字。

def img2vector(filename):
    '''
    用于将存放在txt文件中的以0-1表示的图片转换成一个vector。
    输入filename是字符串格式的文件名；
    输出是一个np.array格式的向量。
    '''
    with open(filename,'r') as f:
        line=f.readline();
        colNum=len(line.strip())
        f.seek(0)
        rowNum=len(list(f))

    vector=np.zeros((rowNum,colNum))
    with open(filename,'r') as f:
        for i in range(rowNum):
            line=f.readline().strip()
            for j in range(colNum):
                vector[i,j]=int(line[j])
    vector=vector.reshape((1,rowNum*colNum))
    return vector



def dir2array(dir):
    '''
    这个函数是我自己写的，用于将trainingDigits和testDigits文件夹中的文本格式数据合并成一个np.array。
    输入dir是文件夹名称，并且要求 dir 在当前目录下；
    返回值有两个：
    data是 dir文件夹中 txt 文件的所有内容
    labels 是对应的类标签
    '''
    fileNames=os.listdir(dir)
    rowNum=len(fileNames)
    #上面得到的文件名称不在当前目录下，所以不能打开，还需要指定路径，
    #由于文件夹 dir 在当前目录下，所以只需要在文件名前加上这个路径就可以了
    filePath=[dir+'/'+s for s in fileNames]
    colNum=img2vector(filePath[0]).shape[1]

    data=np.zeros((rowNum,colNum))
    for i in range(rowNum):
        data[i]=img2vector(filePath[i])

    labels=[s[0] for s in fileNames]
    return data,labels


def handwritingClassify(trainData,trainLabels,testData,testLabels,k):
    '''
    使用上述的knn函数来识别手写数字，
    trainDir是包含训练数据的文件夹名称；
    testDir是包含测试数据的文件夹名称；
    k是KNN的参数。
    '''
    errorCount=0
    for i in range(testData.shape[0]):
        result=knn(testData[i],trainData,trainLabels,3)
        if (result!=testLabels[i]):
            errorCount+=1
    errorRate=errorCount/float(testData.shape[0])
    print("The total errorCount is:",errorCount)
    return errorRate




