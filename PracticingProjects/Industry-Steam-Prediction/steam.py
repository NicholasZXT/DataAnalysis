import os
import numpy as np
import pandas as pd
from typing import List
from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
from sklearn.linear_model import LinearRegression

# %% 数据加载
# DATA_DIR = r"D:\Project-Workspace\Python-Projects\DataAnalysis\local-datasets\工业蒸汽量预测"
DATA_DIR = r"C:\Users\Drivi\Python-Projects\DataAnalysis\local-datasets\工业蒸汽量预测"
train_data = pd.read_csv(os.path.join(DATA_DIR, 'zhengqi_train.txt'), delimiter='\t', header=0)
test_data = pd.read_csv(os.path.join(DATA_DIR, 'zhengqi_test.txt'), delimiter='\t', header=0)
X = train_data.drop(columns=['target'])
y = train_data['target'].copy()

# ***************** EDA ****************
# %% 检查数据缺失情况
X.info()
X_desc = X.describe()
test_data.info()
test_data_desc = test_data.describe()
# 数据很工整，没有缺失值，特征都是连续型


# %% 单变量分析，检查每个特征的分布，是否有缺失值，是否为偏态分布
# 这里使用seaborn绘图，需要将数据转成long-format，每5个变量一组绘图展示
def X_split(df: pd.DataFrame, split_num: int = 5):
    col_num = df.shape[1]
    splits = col_num // split_num
    dfs = []
    for i in range(splits):
        start = i * split_num
        df_part = pd.melt(X, value_vars=['V'+str(i) for i in range(start, start+split_num)], var_name='V')
        dfs.append(df_part)
    final_start = splits * split_num
    df_part = pd.melt(X, value_vars=['V'+str(i) for i in range(final_start, col_num)], var_name='V')
    dfs.append(df_part)
    return dfs

def X_plot(dfs: List[pd.DataFrame]):
    num = len(dfs)
    figs = []
    for i in range(num):
        fig = plt.figure(num=i+1, figsize=(8.0, 5.0), dpi=120)
        ax = fig.add_subplot(111)
        # sns.stripplot(data=dfs[i], x='value', y='V', ax=ax)
        sns.boxplot(data=dfs[i], x='value', y='V', ax=ax)
        # sns.boxenplot(data=dfs[i], x='value', y='V', ax=ax)
        # sns.violinplot(data=dfs[i], x='value', y='V', ax=ax)
        figs.append(fig)
        # fig.show()
        # fig.clear()
    return figs

# 着重检查某个变量的图形
def X_single_check(df: pd.DataFrame, V: str, show=True):
    fig = plt.figure(figsize=(8.0, 6.0), dpi=120)
    ax1 = fig.add_subplot(211)
    sns.boxplot(data=df, x=V, ax=ax1)
    ax2 = fig.add_subplot(212)
    sns.violinplot(data=df, x=V, ax=ax2)
    if show:
        fig.show()
    return fig

dfs = X_split(X)
figs = X_plot(dfs)
# for循环在pycharm中只能展示出4张图，所以只能手动显示
figs[0].show()
figs[1].show()
figs[2].show()
figs[3].show()
figs[4].show()
figs[5].show()
figs[6].show()
figs[7].show()
fig_single = X_single_check(X, 'V9')
# fig_single.show()
# 检查结果
# V9 的量纲范围比较大，有几个异常值需要剔除
# V10, V12, V15, V16, V17, V19, V20,    也要异常值需要剔除
# V23, V25, V30, V31, V33, V34, V35   的分布过于离散，超出 IQR 的样本比较多
# V28, V29, V36, 也需要考虑

#  使用IQR作为异常值判断准则，检查每个特征有多少个异常值
# IQR 的阈值，默认1.5
iqr_th = 3.0
X_q1 = X.quantile(q=0.25)
X_q3 = X.quantile(q=0.75)
X_iqr = X_q3 - X_q1
X_iqr_min = X_q1 - iqr_th * X_iqr
X_iqr_max = X_q3 + iqr_th * X_iqr
iqr_min_flag = X < X_iqr_min
iqr_max_flag = X > X_iqr_max
X_iqr_outlier = iqr_max_flag | iqr_min_flag
# 得到每个特征的异常值数量
X_col_outlier_num = X_iqr_outlier.sum(axis=0)
# 可以看出 V9, V23, V34 的异常值（1.5阈值下）都比较多：835, 700, 488
# 阈值增大到 1.8，V9, V23, V34 的异常值为：833, 678, 450
# 这里为了避免删除太多观测点，IQR阈值放宽到了 3.0 !!!
X_delete_rows = X_iqr_outlier.sum(axis=1) > 0
# 查看需要删除的样本数量，阈值3.0下，需要删除 771 个样本
print(X_delete_rows.sum())
# 删除异常值的样本
X_no_outlier = X.loc[~X_delete_rows, :]
# 检查验证一遍
# t = (X_no_outlier < X_iqr_min) | (X_no_outlier > X_iqr_max)
