# -*- coding: utf-8 -*-
"""
K近邻算法

这里距离使用欧几里得距离，未使用kd树来简化距离对的计算
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
#import operator

n=100
def CreateDataSet(n):
    mu1,mu2=[1,1],[4,4]
    sigma=np.identity(2)
#    多元正态分布随机数产生
    np.random.seed(1)
    group1=np.random.multivariate_normal(mu1,sigma,n)
    group2=np.random.multivariate_normal(mu2,sigma,n)
#    合并两个narray，注意，参数必须要用括号括起来。
    group=np.concatenate((group1,group2))
    labels=[1 for i in range(n)]+[0 for i in range(n)]
    return group,labels

group,labels=CreateDataSet(n)

K=7
observation=[4.5,0.5]

def KNN(obs,group,labels,K):
    size=group.shape[0]
    distance=list()
    for i in range(size):
        distance.append(sqrt((group[i,0]-obs[0])**2+(group[i,1]-obs[1])**2))
    distance=np.array(distance)
    #ndarry可以直接返回排序后的索引，这个很方便。
    K_nearst=distance.argsort()[:K]   
    K_labels=np.array([labels[x] for x in K_nearst])
    print('K_nearst_loc,K_labels,sum=',K_nearst,K_labels,np.sum(K_labels))
    if np.sum(K_labels)>=len(K_labels)/2:
        return 1
    else:
        return 0
    
lab=KNN(observation,group,labels,K)
print('labels of the observation is',lab)

color=['b' for i in range(n)]+['y' for i in range(n)]
#fig=plt.scatter(group[:,0],group[:,1],c=color)
#plot绘图的话，可以一次性绘制多个图像
fig=plt.plot(group[:n,0],group[:n,1],'bo',group[n:,0],group[n:,1],'yo',
             observation[0],observation[1],'r*')
plt.legend(('class1=1','class2=0','observation'))
plt.show()