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
from sklearn.metrics import mean_squared_error as MSE, make_scorer

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
kde_deleted_cols = ['V5', 'V6', 'V9', 'V11', 'V17', 'V22', 'V23']


# %% 计算训练集中各个特征和目标变量的相关性，绘制相关性热力图
# 这里还绘制了各个特征之间的相关性
corr_matrix = pd.concat([X_filter_outlier, y_filter_outlier], axis=1).corr()
fig_corr = plt.figure(figsize=(14, 12), layout='tight')
ax = fig_corr.add_subplot(111)
sns.heatmap(data=corr_matrix, ax=ax, cmap='crest')
# fig_corr.show()


# %% 进行Min-Max的缩放处理，缩放到 [1, 2] 区间
min_max_scaler = MinMaxScaler(feature_range=(1, 2))
min_max_scaler.fit(X_filter_outlier)
# min_max_scaler.data_min_
# min_max_scaler.data_max_
# min_max_scaler.data_range_
X_scale = min_max_scaler.transform(X_filter_outlier)
X_scale = pd.DataFrame(X_scale, columns=X_filter_outlier.columns)
X_test_scale = min_max_scaler.transform(test_data)
X_test_scale = pd.DataFrame(X_test_scale, columns=test_data.columns)
# 观察一下缩放前后特征的kde分布，可以看出，min-max 缩放并没有改变分布
# fig1 = skew_single_check(X_filter_outlier, 'V0', figlabel='before')
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
# 观察一下变换前后的变量分布
# fig1 = skew_single_check(X_scale, 'V0', figlabel='before')
# fig2 = skew_single_check(X_boxcox, 'V0', figlabel='after')
# fig1.show()
# fig2.show()

# ***************** 特征选择 ******************
# 特征选择放到最后再做，因为一般来说，特征选择都会降低模型的效果
# %% 根据KDE的分析，从训练集和测试集中删除分布不一致的特征
kde_used_cols = [v for v in X_boxcox.columns if v not in kde_deleted_cols]
X_kde_filter = X_boxcox[kde_used_cols].copy()
X_test_kde_filter = X_test_box[kde_used_cols].copy()

# %% 按照和目标变量的相关性来过滤特征
corr_threshold = 0.5
# 只取目标变量和各个特征的相关性这一列
corr_cols = corr_matrix['target'].iloc[:-1]
corr_cols_filter = corr_cols[corr_cols.abs() >= corr_threshold]
corr_used_cols = list(corr_cols_filter.index)
print('corr_used_cols: ', corr_used_cols)
X_corr_filter = X_boxcox[corr_used_cols].copy()
X_test_corr_filter = X_test_box[corr_used_cols].copy()

# %% 同时做特征的 kde过滤 和 相关性过滤
kde_corr_used_cols = list(set(kde_used_cols) & set(corr_used_cols))
print('kde_corr_used_cols: ', kde_corr_used_cols)
X_cols_filter = X_boxcox[kde_corr_used_cols].copy()
X_test_cols_filter = X_test_box[kde_corr_used_cols].copy()

# ********************* 模型训练 *************************
# %% 使用线性回归模型作为基准模型
random_state = 29
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
train_split_args = {'train_size': 0.8, 'shuffle': True, 'random_state': random_state}
mse_scorer = make_scorer(MSE)
def lr_model_compute(X, y):
    lr = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, **train_split_args)
    lr.fit(X_train, y_train)
    # sklearn的LinearRegression的score方法，默认返回的是 R^2
    cv_score = cross_val_score(lr, X, y, cv=kf)
    # 再计算一下 MSE 这个指标
    cv_mse = cross_val_score(lr, X, y, cv=kf, scoring=mse_scorer)
    res_infos = ['train_score', 'test_score', 'cv_score_mean', 'cv_score_std', 'train_mse', 'test_mse', 'cv_mse_mean', 'cv_mse_std']
    res = [lr.score(X_train, y_train), lr.score(X_test, y_test), cv_score.mean(), cv_score.std(),
           MSE(y_train, lr.predict(X_train)), MSE(y_test, lr.predict(X_test)), cv_mse.mean(), cv_mse.std()]
    return lr, res_infos, res
# 先使用 只做了异常值过滤 的数据进行建模
lr_naive, res_infos, lr_naive_res = lr_model_compute(X_filter_outlier, y_filter_outlier)
# 使用经过 异常值过滤 + 特征缩放 + Box-Cox变换 的数据进行建模
lr_boxcox, _, lr_boxcox_res = lr_model_compute(X_boxcox, y_filter_outlier)
# 使用经过 异常值过滤 + 特征缩放 + Box-Cox变换 + kde特征过滤 的数据进行建模
lr_kde, _, lr_kde_res = lr_model_compute(X_kde_filter, y_filter_outlier)
# 使用经过 异常值过滤 + 特征缩放 + Box-Cox变换 + 相关性特征过滤 的数据进行建模
lr_corr, _, lr_corr_res = lr_model_compute(X_corr_filter, y_filter_outlier)
# 使用经过 异常值过滤 + 特征缩放 + Box-Cox变换 + kde特征过滤 + 相关性特征过滤 的数据进行建模
lr_filter, _, lr_filter_res = lr_model_compute(X_cols_filter, y_filter_outlier)
index = ['lr_naive', 'lr_boxcox', 'lr_kde', 'lr_corr', 'lr_filter']
lr_res = [lr_naive_res, lr_boxcox_res, lr_kde_res, lr_corr_res, lr_filter_res]
lr_res_df = pd.DataFrame(data=lr_res, columns=res_infos, index=index)
print(lr_res_df)
# 从结果来看，从 异常值过滤 --> (特征缩放 + Box-Cox变换) --> {kde特征过滤, 相关性特征过滤} 这一套特征工程下来，不论是 R^2 还是 MSE
# 准确性都是逐渐降低的，这并不奇怪，因为特征选择就是这样。
# lr_naive --> lr_boxcox 的准确性略有下降，暂不清楚为啥
# lr_boxcox --> lr_kde 的准确性基本没有下降，说明 kde 过滤掉的特征是无关紧要的
# lr_boxcox --> lr_corr 的准确性下降了一些，说明相关性过滤掉的特征里，有部分是有用的
# lr_corr 约等于 lr_filter 的准确度，再次说明 kde 过滤的特征确实是无关紧要的

# %% 绘制线性基准模型的残差图
y_pred_naive = lr_naive.predict(X_filter_outlier)
y_err_naive = y_filter_outlier - y_pred_naive
y_pred_box = lr_boxcox.predict(X_boxcox)
y_err_box = y_filter_outlier - y_pred_box
y_pred_filter = lr_filter.predict(X_cols_filter)
y_err_filter = y_filter_outlier - y_pred_filter
# 绘图
fig = plt.figure(figsize=(16, 12), layout='tight')
ax = fig.add_subplot(2, 2, 1)
sns.scatterplot(x=y_pred_naive, y=y_err_naive, ax=ax)
ax.set_title('lr_naive residual plot')
ax.set_xlabel('fitted value')
ax.set_ylabel('residual')
ax = fig.add_subplot(2, 2, 2)
sns.scatterplot(x=y_pred_box, y=y_err_box, ax=ax)
ax.set_title('lr_boxcox residual plot')
ax.set_xlabel('fitted value')
ax.set_ylabel('residual')
ax = fig.add_subplot(2, 2, 3)
sns.scatterplot(x=y_pred_filter, y=y_err_filter, ax=ax)
ax.set_title('lr_filter residual plot')
ax.set_xlabel('fitted value')
ax.set_ylabel('residual')
fig.suptitle('residual plots for linear regression models')
# fig.show()
# 残差图上可以看出：1.数据中存在线性关系（残差图没有明显的趋势关系）；2.误差项似乎并不等方差（分布并不等宽）；3.存在个别异常值

# %% 绘制线性基准模型的验证曲线和学习曲线
# 对于LR来说，没有超参数，所以不需要绘制验证曲线，这里只绘制学习曲线
train_sizes = np.linspace(0.1, 1, 10)
lr_curve_args = {'train_sizes': train_sizes, 'shuffle': True, 'random_state': random_state, 'cv': kf, 'scoring': mse_scorer}
train_size_abs, train_scores, test_scores = learning_curve(lr_filter, X_cols_filter, y_filter_outlier, **lr_curve_args)
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)
# 绘图
fig_lr = plt.figure()
ax = fig_lr.add_subplot(111)
sns.lineplot(x=train_size_abs, y=train_scores_mean, ax=ax, color='blue', label='train_scores_mean')
sns.lineplot(x=train_size_abs, y=test_scores_mean, ax=ax, color='red', label='test_scores_mean')
ax.legend()
# fig_lr.suptitle("learning curve for linear regression")
ax.set_title("Learning curve for linear regression")
fig_lr.show()
# 这个学习曲线差不多能看出，对于线性模型来说，已经是极限了，再增加样本也提高不了多少了

# %% 使用线性基准模型进行预测，查看效果
y_test_pred_naive = lr_naive.predict(test_data)
y_test_pred_box = lr_boxcox.predict(X_test_box)
y_test_pred_filter = lr_filter.predict(X_test_cols_filter)
y_test_pred_df = pd.DataFrame({'lr_naive': y_test_pred_naive, 'lr_boxcox': y_test_pred_box, 'lr_filter': y_test_pred_filter})
y_test_pred_df[['lr_naive']].to_csv('steam_predict_lr_naive.txt', header=False, index=False)
y_test_pred_df[['lr_boxcox']].to_csv('steam_predict_lr_boxcox.txt', header=False, index=False)
y_test_pred_df[['lr_filter']].to_csv('steam_predict_lr_filter.txt', header=False, index=False)
# 天池测试集结果
# lr_naive: 2.8813,  lr_filter: 0.4673
# 这里可以看出，虽然在训练集上 lr_naive 和 lr_filter 上看不出差别，甚至 lr_naive 要好一点，但是在测试集上的泛化性能差异很大
# 说明这里估计泛化误差的方式有问题？？
# 应该不是泛化误差的估计方式有问题，而是因为这里的训练集（被划分的train和test都属于这个训练集）中，部分特征的分布和待预测的测试集中差异很大——就
# 像kde图展示的那样，导致 lr_naive 在待预测数据上的泛化性能与训练时的表现差异很大，而 lr_filter 由于过滤掉了这些特征，所以受到的影响没有那么大