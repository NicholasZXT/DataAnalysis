# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 16:22:11 2018
K-means Clustering

@author: xtzhang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

K=3
N=50
def CreateDataSet(n):
    mu1,mu2,mu3=[1,1],[9,1],[5,6]
    sigma=np.identity(2)
    D1=np.random.multivariate_normal(mu1,sigma,n)
    D2=np.random.multivariate_normal(mu2,sigma,n)
    D3=np.random.multivariate_normal(mu3,sigma,n)
    data=np.concatenate((D1,D2,D3))
    return pd.DataFrame(data)

data=CreateDataSet(N)

def InitCentroid(data,K):
    n=data.shape[0]
    np.random.seed(1)
    labels=np.random.randint(K,size=n)
    data=pd.DataFrame(data,index=labels)
    centroid=list()
    for i in range(K):
        temp=data.loc[i]
#        print 'temp.mean=',temp.mean()
        centroid.append(temp.mean())
    return np.array(centroid)

def NearstCentroid(data,centroid):
    # n=data.shape[0]
    K=centroid.shape[0]
    labels=list()
    for  index,obs in data.iterrows():
        dis=[((obs[0]-centroid[i,0])**2+(obs[1]-centroid[i,1])**2)**0.5 for i in range(K)]
        dis=np.array(dis)
        labels.append(dis.argmin())
    newdata=pd.DataFrame(data.values,index=labels)
    return newdata


def NewCentroid(data,K):
    centroid=list()
    for i in range(K):
        temp=data.loc[i]
        centroid.append(temp.mean())
    return np.array(centroid)



def K_Means(data,K):
    datasize=data.shape[0]
    centroid=InitCentroid(data,K)
    for i in range(600):
        data=NearstCentroid(data,centroid)
        centroid=NewCentroid(data,K)
    return centroid

centroid=K_Means(data,K)

fig=plt.plot(data.iloc[:,0],data.iloc[:,1],'bo',centroid[:,0],centroid[:,1],'r+')
plt.show()
print("the centroids are\n",centroid)
    
    