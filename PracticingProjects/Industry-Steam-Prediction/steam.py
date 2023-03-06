import os
from typing import List
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve, validation_curve
from sklearn.metrics import mean_squared_error

# %% 数据加载
DATA_DIR = r"D:\Project-Workspace\Python-Projects\DataAnalysis\local-datasets\工业蒸汽量预测"
# DATA_DIR = r"C:\Users\Drivi\Python-Projects\DataAnalysis\local-datasets\工业蒸汽量预测"
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

# %% 检查每个特征的分布，绘制每个特征的箱线图
# 多个特征的箱线图放到同一个图形里的方式不太好，因为各个特征的量纲会相互影响，改为采用每个特征一个子图的方式绘制，保留每个特征自己的量纲
def box_single_check(df: pd.DataFrame, V: str, show=True):
    """
    着重检查某个变量的箱线图和小提琴图
    """
    fig = plt.figure(figsize=(8.0, 6.0), dpi=120)
    ax1 = fig.add_subplot(211)
    sns.boxplot(data=df, x=V, ax=ax1)
    ax2 = fig.add_subplot(212)
    sns.violinplot(data=df, x=V, ax=ax2)
    if show:
        fig.show()
    return fig

def box_subplot(df: pd.DataFrame, rows: int = 4, cols: int = 2, w=6.8, h=4.8):
    """
    每个变量一个子图，每个Figure绘制一个 rows x cols 布局的箱线图
    """
    figsize = (w, h)
    col_num = df.shape[1]
    col_num_per_fig = rows * cols
    fig_num = col_num // col_num_per_fig
    figs = []
    for i in range(fig_num):
        # 注意设置 clear=True， 否则可能拿到的是已经使用过的Figure
        fig = plt.figure(num='box-'+str(i+1), figsize=figsize, dpi=100, clear=True, layout='tight')
        for j in range(col_num_per_fig):
            feature_num = i * col_num_per_fig + j
            feature = 'V' + str(feature_num)
            print(f"drawing subplot [{i+1}] with shape '({rows}, {cols})' for feature '{feature}'.")
            ax = fig.add_subplot(rows, cols, j+1)
            sns.boxplot(data=df, x=feature, ax=ax)
            ax.set_ylabel(feature, rotation=0)
            ax.set_xlabel('')
            # ax.xaxis.set_visible(False)
        figs.append(fig)
    cols_remain = col_num % col_num_per_fig
    final_rows = int(np.ceil(cols_remain / cols))
    final_fig = plt.figure(num='box-'+str(fig_num+1), figsize=figsize, dpi=100, clear=True, layout='tight')
    # print(f"col_num: {col_num}, cols_remain: {cols_remain}")
    for j in range(cols_remain):
        feature_num = col_num - cols_remain + j
        feature = 'V' + str(feature_num)
        print(f"drawing subplot [{fig_num+1}] with shape '({final_rows}, {cols})' for feature '{feature}'.")
        ax = final_fig.add_subplot(final_rows, cols, j+1)
        sns.boxplot(data=df, x=feature, ax=ax)
        ax.set_ylabel(feature, rotation=0)
        ax.set_xlabel('')
        # ax.xaxis.set_visible(False)
    figs.append(final_fig)
    return figs

# %% 分析每个特征的箱线图
box_figs = box_subplot(df=X, rows=4, cols=3, w=16, h=10)
box_figs[0].show()
box_figs[1].show()
box_figs[2].show()
box_figs[3].show()
box_figs[4].show()
# 着重检查某个变量
box_fig_single = box_single_check(X, 'V36')
# fig_single.show()
# 检查结果
# V9 的量纲范围比较大，有几个异常值需要剔除
# V10, V12, V15, V16, V17, V19, V20 有异常值需要剔除
# V23, V25, V30, V31, V33, V34, V35   的分布过于离散，超出 IQR 的样本比较多
# V28, V29, V36, 也需要考虑

# %% 使用IQR作为异常值判断准则，检查每个特征有多少个异常值，进行异常值过滤
# IQR 的阈值，默认1.5
# 这里为了避免删除太多样本点，IQR阈值放宽到了 3.0 !!!
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
col_outlier_num = X_iqr_outlier.sum(axis=0)
# 可以看出 V9, V23, V34 的异常值（1.5阈值下）都比较多：835, 700, 488
# 阈值增大到 1.8，V9, V23, V34 的异常值为：833, 678, 450
iqr_deleted_rows = X_iqr_outlier.sum(axis=1) > 0
# 查看需要删除的样本数量，阈值3.0下，需要删除 771 个样本
print(iqr_deleted_rows.sum())
# 删除异常值的样本
X_filter_outlier = X.loc[~iqr_deleted_rows, :]
y_filter_outlier = y.loc[~iqr_deleted_rows]
# 检查验证一遍
# t = (X_filter_outlier < X_iqr_min) | (X_filter_outlier > X_iqr_max)


# %% 绘制每个特征的直方图和Q-Q图，查看分布的偏度
def skew_subplot(df, cols=4, w=6.4, h=4.8):
    """
    对每个特征绘制直方图和Q-Q图，以上、下两行的形式放在同一列，cols指定的是每一幅图中绘制的特征个数
    """
    figsize = (w, h)
    col_num = df.shape[1]
    fig_num = col_num // cols
    figs = []
    for i in range(fig_num):
        # 注意设置 clear=True， 否则可能拿到的是上面已经使用过的Figure
        fig = plt.figure(num='skew-'+str(i+1), figsize=figsize, dpi=100, clear=True, layout='tight')
        for j in range(cols):
            feature_num = i * cols + j
            feature_name = 'V' + str(feature_num)
            print(f"drawing subplot [{i + 1}] with shape '(2, {cols})' for feature '{feature_name}'.")
            # 每个特征的直方图放在第一行
            ax1 = fig.add_subplot(2, cols, j+1)
            sns.histplot(x=df[feature_name], ax=ax1)
            # 每个特征的Q-Q图放在第二行
            ax2 = fig.add_subplot(2, cols, j+1+cols)
            osm, osr = stats.probplot(df[feature_name], dist='norm', fit=False)
            sns.scatterplot(x=osm, y=osr, ax=ax2)
            ax1.set_title(feature_name)
            ax1.set_xlabel('')
            ax2.set_xlabel('')
        # 上面使用了 tight，这里可以不必使用了
        # fig.tight_layout()
        figs.append(fig)
    col_remain = col_num % cols
    if col_remain == 0:
        return figs
    final_fig = plt.figure(num='skew-'+str(fig_num+1), figsize=figsize, dpi=100, clear=True, layout='tight')
    for j in range(col_remain):
        feature_num = fig_num * cols + j
        feature_name = 'V' + str(feature_num)
        print(f"drawing subplot [{fig_num + 1}] with shape '(2, {cols})' for feature '{feature_name}'.")
        ax1 = final_fig.add_subplot(2, cols, j+1)
        sns.histplot(x=df[feature_name], ax=ax1)
        ax2 = final_fig.add_subplot(2, cols, j+1+cols)
        osm, osr = stats.probplot(df[feature_name], dist='norm', fit=False)
        sns.scatterplot(x=osm, y=osr, ax=ax2)
        ax1.set_title(feature_name)
        ax1.set_xlabel('')
        ax2.set_xlabel('')
    figs.append(final_fig)
    return figs

def skew_single_check(df, feature,  w=6.4, h=4.8, show=False, figlabel=None):
    """
    绘制指定变量的直方图和Q-Q图
    """
    figsize = (w, h)
    figlabel = figlabel if figlabel else 'skew-single-'+feature
    fig = plt.figure(num=figlabel, figsize=figsize, clear=True)
    ax1 = fig.add_subplot(121)
    sns.histplot(x=df[feature], ax=ax1)
    ax2 = fig.add_subplot(122)
    osm, osr = stats.probplot(df[feature], dist='norm', fit=False)
    sns.scatterplot(x=osm, y=osr, ax=ax2)
    fig.suptitle(feature)
    if show:
        fig.show()
    return fig


# %% 分析各个特征的直方图和Q-Q图
# 使用没有删除异常值的数据绘图的话，也能发现有些特征偏离的比较厉害
# skew_figs = skew_subplot(X, cols=5, w=25, h=10)
# cols_used = ['V'+str(i) for i in range(10)]
# skew_figs = skew_subplot(X_filter_outlier[cols_used], cols=5, w=25, h=10)
skew_figs = skew_subplot(X_filter_outlier, cols=5, w=25, h=10)
skew_figs[0].show()
skew_figs[1].show()
skew_figs[2].show()
skew_figs[3].show()
skew_figs[4].show()
skew_figs[5].show()
skew_figs[6].show()
skew_figs[7].show()
# 检查单个变量
# skew_fig_single = skew_single_check(X, feature='V9', w=10, h=5, show=True)
skew_fig_single = skew_single_check(X_filter_outlier, feature='V22', w=10, h=5, show=True)
# 检查结果为：
# V0, V1, V5, V6, V7, V8, V11, V14, V16, V18 偏离正态较严重，需要后续做变换处理
# V9, V17, V22, V23, V24, V28, V35 要特别注意，似乎取值只有几个离散的值
# V33, V34 这两个变量中间的值有跳变
skew_cols = ['V0', 'V1', 'V5', 'V6', 'V7', 'V8', 'V11', 'V14', 'V16', 'V18']


# %% 对比训练数据和测试数据各个特征的分布情况，绘制KDE图
def kde_subplot(train_df, test_df, rows=4, cols=4, w=6.4, h=4.8):
    figsize = (w, h)
    col_num = train_df.shape[1]
    subplot_num = rows * cols
    fig_num = col_num // subplot_num + 1
    figs = []
    for i in range(fig_num):
        fig = plt.figure(num='kde-'+str(i+1), figsize=figsize, clear=True, layout='tight')
        for j in range(subplot_num):
            feature_num = i * subplot_num + j
            if feature_num >= col_num:
                break
            feature_name = 'V' + str(feature_num)
            print(f"drawing subplot [{i + 1}] with shape '({rows}, {cols})' for feature '{feature_name}'.")
            ax = fig.add_subplot(rows, cols, j+1)
            sns.kdeplot(train_df, x=feature_name, color='red', label='train', ax=ax)
            sns.kdeplot(test_df, x=feature_name, color='blue', label='test', ax=ax)
            ax.legend()
        figs.append(fig)
    return figs

# %% 分析KDE图
kde_figs = kde_subplot(X_filter_outlier, test_data, w=16, h=12)
kde_figs[0].show()
kde_figs[1].show()
kde_figs[2].show()
# 对比之下，可以看出：
# V5, V6, V9, V11, V17, V22, V23 这几个变量，训练集和测试集的分布差异太大了，所以需要剔除掉，不能使用
# V14, V19, V21, V35 这几个变量也略有偏离

# %% 根据KDE的分析，从训练集和测试集中删除分布不一致的特征
kde_deleted_cols = ['V5', 'V6', 'V9', 'V11', 'V17', 'V22', 'V23']
kde_used_cols = [v for v in X_filter_outlier.columns if v not in kde_deleted_cols]
X_kde_filter = X_filter_outlier[kde_used_cols].copy()
X_test_kde_filter = test_data[kde_used_cols].copy()

# %% 计算训练集中剩余特征和目标变量的相关性，绘制相关性热力图
# 这里还绘制了各个特征之间的相关性
corr_matrix = pd.concat([X_kde_filter, y_filter_outlier], axis=1).corr()
fig_corr = plt.figure(figsize=(14, 12), layout='tight')
ax = fig_corr.add_subplot(111)
sns.heatmap(data=corr_matrix, ax=ax, cmap='crest')
fig_corr.show()

# %% 按照和目标变量的相关性来过滤特征
# 相关性阈值
corr_threshold = 0.5
# 只取目标变量和各个特征的相关性这一列
corr_cols = corr_matrix['target'].iloc[:-1]
corr_cols_filter = corr_cols[corr_cols.abs() >= corr_threshold]
corr_used_cols = list(corr_cols_filter.index)
print('corr_used_cols: ', corr_used_cols)
# 进行过滤
X_corr_filter = X_kde_filter[corr_used_cols].copy()
X_test_corr_filter = X_test_kde_filter[corr_used_cols].copy()

# %% 进行Min-Max的缩放处理，缩放到 [1, 2] 区间
min_max_scaler = MinMaxScaler(feature_range=(1, 2))
min_max_scaler.fit(X_corr_filter)
# min_max_scaler.data_min_
# min_max_scaler.data_max_
# min_max_scaler.data_range_
X_scale = min_max_scaler.transform(X_corr_filter)
X_scale = pd.DataFrame(X_scale, columns=X_corr_filter.columns)
X_test_scale = min_max_scaler.transform(X_test_corr_filter)
X_test_scale = pd.DataFrame(X_test_scale, columns=X_test_corr_filter.columns)
# 观察一下缩放前后特征的kde分布，可以看出，min-max 缩放并没有改变分布
# fig1 = skew_single_check(X_corr_filter, 'V0', figlabel='before')
# fig2 = skew_single_check(X_scale, 'V0', figlabel='after')
# fig1.show()
# fig2.show()

# %% 对偏态分布的特征做 Box-Cox 变换
# 这里做 Box-Cox 变换之前必须要做 min-max 缩放，因为原始数据中有负值
def box_cox_transform(df, cols):
    df_transform = df.copy()
    for col in cols:
        col_value, maxlog = stats.boxcox(df[col])
        df_transform[col] = col_value
    return df_transform

skew_cols_to_change = [col for col in skew_cols if col in X_scale.columns]
X_boxcox = box_cox_transform(X_scale, skew_cols_to_change)
X_test_box = box_cox_transform(X_test_scale, skew_cols_to_change)
# 检查一下变换前后的变量分布
fig1 = skew_single_check(X_corr_filter, 'V0', figlabel='before')
fig2 = skew_single_check(X_boxcox, 'V0', figlabel='after')
fig1.show()
fig2.show()

# %% 使用线性回归模型作为基准模型
random_state = 29
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

lr_naive = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_filter_outlier, y_filter_outlier, train_size=0.8, shuffle=True, random_state=random_state)
lr_naive.fit(X_train, y_train)
lr_naive_mse = mean_squared_error(y_test, lr_naive.predict(X_test))
lr_naive_cv_score = cross_val_score(lr_naive, X_filter_outlier, y_filter_outlier, cv=kf)
lr_naive_res = [lr_naive.score(X_train, y_train), lr_naive.score(X_test, y_test), lr_naive_mse,
                lr_naive_cv_score.mean(), lr_naive_cv_score.std()]

lr_corr = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_corr_filter, y_filter_outlier, train_size=0.8, shuffle=True, random_state=random_state)
lr_corr.fit(X_train, y_train)
lr_corr_mse = mean_squared_error(y_test, lr_corr.predict(X_test))
lr_corr_cv_score = cross_val_score(lr_corr, X_corr_filter, y_filter_outlier, cv=kf)
lr_corr_res = [lr_corr.score(X_train, y_train), lr_corr.score(X_test, y_test), lr_corr_mse,
               lr_corr_cv_score.mean(), lr_corr_cv_score.std()]

lr_box = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_boxcox, y_filter_outlier, train_size=0.8, shuffle=True, random_state=random_state)
lr_box.fit(X_train, y_train)
lr_box_mse = mean_squared_error(y_test, lr_box.predict(X_test))
lr_box_cv_score = cross_val_score(lr_box, X_boxcox, y_filter_outlier, cv=kf)
lr_box_res = [lr_box.score(X_train, y_train), lr_box.score(X_test, y_test), lr_box_mse,
              lr_box_cv_score.mean(), lr_box_cv_score.std()]

score_cols = ['train_score', 'test_score', 'test_mse', 'cv_score_mean', 'cv_score_std']
index = ['lr_naive', 'lr_corr', 'lr_box']
lr_score_df = pd.DataFrame(data=[lr_naive_res, lr_corr_res, lr_box_res], columns=score_cols, index=index)
print(lr_score_df)

# 从结果来看，这一套特征工程下来，结果还不如只是做了异常值过滤的数据效果好
# 但是也不能这么说，因为后面两个只用了9个特征，lr_naive 用了 38 个特征，
