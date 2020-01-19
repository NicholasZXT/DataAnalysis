
import numpy as np
import pandas as pd


def calSimilarity(data, user, item, simMethod):
    """
    用于在给定相似度下，计算user对于item的评分
    :param data: 用户和物品的评价矩阵，行对应于user,列对应于item
    :param user: user的index，对应于行
    :param item: item的index，对应于列
    :param simMethod: 相似度的计算方式
    :return:
    """
    # 得到物品的数量，也就是data中的列数
    m = data.shape[1]

    for j in range(m):
        userRating = data[user, j]
        if userRating == 0:
            continue



def recommend(data, user, N = 3, simMethod):
    pass