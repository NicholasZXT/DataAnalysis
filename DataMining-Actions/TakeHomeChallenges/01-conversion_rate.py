# %% 导入必要包
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score,  precision_score, recall_score, confusion_matrix, roc_curve, classification_report
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# %% 读取数据
base_path = os.path.join(os.getcwd(), r'dataset/ds_takehome_challenges')
data_file = r'01. conversion_project.csv'
data_path = os.path.join(base_path, data_file)
print(os.path.exists(data_path))
data = pd.read_csv(data_path, header=0)

# %% 检查数据
# data.columns.tolist()
# ['country', 'age', 'new_user', 'source', 'total_pages_visited', 'converted']
# data['country'].unique().tolist()
# ['UK', 'US', 'China', 'Germany']
# data['source'].unique().tolist()
# ['Ads', 'Seo', 'Direct']
# data['age'].describe()
# count    316200.000000
# mean         30.569858
# std           8.271802
# min          17.000000
# 25%          24.000000
# 50%          30.000000
# 75%          36.000000
# max         123.000000
# Name: age, dtype: float64
# data['total_pages_visited'].describe()
# count    316200.000000
# mean          4.872966
# std           3.341104
# min           1.000000
# 25%           2.000000
# 50%           4.000000
# 75%           7.000000
# max          29.000000
# Name: total_pages_visited, dtype: float64
cols = ['country', 'age', 'new_user', 'source', 'total_pages_visited']
X = data[cols]
y = data['converted']
# 划分成 （训练集+验证集）+测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=6200, random_state=29, stratify=y)
# 划分一次训练集+验证集
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=31000, random_state=29, stratify=y_trainval)
X_train.reset_index(inplace=True, drop=True)
X_val.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
y_val.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)
# 检查比例
# y.sum()/y.shape[0]
# y_test.sum()/y_test.shape[0]
# 缺失值检查——无缺失值
# X_train.isna().sum()
# X_test.isna().sum()


# %% 手动特征工程
# 对 country 和 source 进行 one-hot 编码
onehot = OneHotEncoder(drop='first', sparse=False, handle_unknown='error')
onehot_cols = ['country', 'source']
onehot.fit(X_train[onehot_cols])
# onehot.categories_
# onehot.drop_idx_
onehot_cols_res = ['country_' + v for v in onehot.categories_[0].tolist()[1:]] +\
              ['source_' + v for v in onehot.categories_[1].tolist()[1:]]
X_train_p1 = onehot.transform(X_train[onehot_cols])
X_test_p1 = onehot.transform(X_test[onehot_cols])
X_train_p1 = pd.DataFrame(X_train_p1, columns=onehot_cols_res)
X_test_p1 = pd.DataFrame(X_test_p1, columns=onehot_cols_res)

# 对 age 和 total_pages_visited 进行分箱
kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
kbin_cols = ['age', 'total_pages_visited']
kbin.fit(X_train[kbin_cols])
# kbin.bin_edges_
X_train_p2 = kbin.transform(X_train[kbin_cols]) + 1
X_test_p2 = kbin.transform(X_test[kbin_cols]) + 1
X_train_p2 = pd.DataFrame(X_train_p2, columns=kbin_cols)
X_test_p2 = pd.DataFrame(X_test_p2, columns=kbin_cols)

# 合并特征
X_train_ = pd.concat([X_train_p1, X_train_p2, X_train[['new_user']]], axis=1)
X_test_ = pd.concat([X_test_p1, X_test_p2, X_test[['new_user']]], axis=1)


# %% 构造特征工程流水线
# 自定义特征转换器，实现对 country 和 source 进行 one-hot 编码，对 age 和 total_pages_visited 进行分箱的处理，方便后续使用pipeline
class FeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, onehot_transformer, onehot_cols, kbin_transformer, kbin_cols, other_cols):
        self.onehot = onehot_transformer
        self.onehot_cols = onehot_cols
        self.kbin = kbin_transformer
        self.kbin_cols = kbin_cols
        self.other_cols = other_cols
        # 使用 ColumnTransformer 来对不同的列进行不同的特征处理
        transformers = [(onehot_transformer, onehot_cols), (kbin_transformer, kbin_cols)]
        self.cols_transformer = make_column_transformer(*transformers, remainder='passthrough')
        self.onehot_cols_res = []

    def make_onehot_cols_res(self):
        # 拼凑出one-hot编码之后的特征名称
        onehot = self.cols_transformer.named_transformers_['onehotencoder']
        onehot_drop_method = onehot.get_params()['drop']
        if onehot_drop_method == 'first':
            for idx, col in enumerate(onehot_cols):
                self.onehot_cols_res.extend([col + '_' + v for v in onehot.categories_[idx].tolist()[1:]])
        else:
            for idx, col in enumerate(onehot_cols):
                self.onehot_cols_res.extend([col + '_' + v for v in onehot.categories_[idx].tolist()])

    def fit(self, X):
        self.cols_transformer.fit(X)
        self.make_onehot_cols_res()

    def transform(self, X):
        X_ = self.cols_transformer.transform(X)
        # return X_
        X_cols = self.onehot_cols_res + self.kbin_cols + self.other_cols
        X = pd.DataFrame(X_, columns=X_cols)
        # 分箱特征的初始值设为从1开始，而不是从0开始
        X[self.kbin_cols] = X[self.kbin_cols] + 1
        return X


onehot = OneHotEncoder(drop='first', sparse=False, handle_unknown='error')
onehot_cols = ['country', 'source']
kbin = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
kbin_cols = ['age', 'total_pages_visited']
other_cols = ['new_user']
features_transformer = FeaturesTransformer(onehot, onehot_cols, kbin, kbin_cols, other_cols)
features_transformer.fit(X_train)
X_train_enc = features_transformer.transform(X_train)
# features_transformer.cols_transformer
# t = features_transformer.cols_transformer.transform(X_train)
# 对比手动特征工程的结果
# t = X_train_ - X_train_enc
# t.sum()


# %% logistic regression 基本建模
# 看下模型的基准结果
lr = LogisticRegression()
lr.fit(X_train_enc, y_train)
# lr.coef_
# lr.intercept_
y_train_pred = lr.predict(X_train_enc)
# 准确率
# lr.score(X_train_enc, y_train)
print("logistic regression accuracy: ", accuracy_score(y_train, y_train_pred))
# 0.9841
# 精确率
print("logistic regression precision of positive: ", precision_score(y_train, y_train_pred))
# 0.8417576492788785
# pos_label=0，计算的就是负类的 precision
print("logistic regression precision of negative: ", precision_score(y_train, y_train_pred, pos_label=0))
# 0.9875900998410343
# 召回率
print("logistic regression recall: ", recall_score(y_train, y_train_pred))
# 0.6245
# pos_label=0，计算的就是 specifity，也就是实际负类的样本中预测正确的比例
print("logistic regression specifity: ", recall_score(y_train, y_train_pred, pos_label=0))
# 0.9960866666666667
# 混淆矩阵
labels = [1, 0]
print(f"logistic regression confusion_matrix of labels {labels}:")
print(confusion_matrix(y_train, y_train_pred, labels=labels))
# tp, fn, fp, tn = confusion_matrix(y_train, y_train_pred, labels=labels).ravel()
# array([[  6245,   3755],
#        [  1174, 298826]])
# 分类报告
print(f"logistic regression classification_report of labels {labels}:")
print(classification_report(y_train, y_train_pred, labels=labels, target_names=['正类', '负类']))


# %% logistic regression 超参数搜索建模


# %% svm 模型


# %% gbdt 模型


# %% xbgoost 模型