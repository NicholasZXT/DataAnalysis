import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV,train_test_split


#查看数据
# %cd Kaggle/Titanic/
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# describe只会给出数值型变量的统计信息，没有多少用处
# train.describe()
# info()可以看出，age, cabin, embarked 含有缺失值
# train.info()
# test.info()
# 将乘客ID设为index，这个变量对于预测没有用
train.set_index('PassengerId', inplace=True)
test.set_index('PassengerId', inplace=True)


# -----------------------缺失值处理-----------------------
# 查看缺失值个数以及占比
train.isnull().sum().sort_values(ascending=False)
(train.isnull().sum()/train.isnull().count()*100).sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
(test.isnull().sum()/test.isnull().count()*100).sort_values(ascending=False)
train['Embarked'].value_counts()
train['Embarked'].mode()[0]
# 填充缺失值
# Embarked变量
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)
test['Fare'].fillna(test['Fare'].median(), inplace = True)
# Cabin变量
train.drop(['Cabin'], axis=1, inplace=True)
test.drop(['Cabin'], axis=1, inplace=True)
# Age变量
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)
# 最后检查是否有缺失值
# train.isnull().sum().sort_values(ascending=False)
# test.isnull().sum().sort_values(ascending=False)

y = train['Survived']
X = train.drop(['Survived'], axis=1)

# ---------------特征工程----------------------------------

# 利用SibSp和Parch这两个特征构建一个新的特征FamilySize
X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

# 对乘客姓名进行处理,提取乘客的Title
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# t = X['Name'].iloc[0]
# t2 = re.search('([A-Za-z]+)\.', t)
# get_title(t)
X['Title'] = X['Name'].apply(get_title)
test['Title'] = test['Name'].apply(get_title)
# 将Title进行改写转换
for dataset in [X, test]:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don',
                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# 对年龄Age进行分箱处理，而不是直接作为数值
for dataset in [X, test]:
    dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])

# 对费用Fare也进行分箱处理，
for dataset in [X, test]:
    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare',
                                                                                      'Average_fare','high_fare'])

# 最后丢弃不再需要的特征
# 这里不知道为啥不丢弃SibSp和Parch
for dataset in [X, test]:
    drop_column = ['Age','Fare','Name','Ticket']
    dataset.drop(drop_column, axis=1, inplace = True)

# 对离散特征进行处理
X_dummy = pd.get_dummies(X, columns=["Sex","Title","Age_bin","Embarked","Fare_bin"],
                   prefix=["Sex","Title","Age_type","Em_type","Fare_type"])
test_dummy = pd.get_dummies(test, columns=["Sex","Title","Age_bin","Embarked","Fare_bin"],
                   prefix=["Sex","Title","Age_type","Em_type","Fare_type"])

# 绘制变量之间的相关系数矩阵
plt.style.use("ggplot")
sns.set()
sns.heatmap(X_dummy.corr(), annot=True, linewidths=0.2)

# ------------训练模型------------------------------
# LogisticRegression
logis = LogisticRegression()
logis.fit(X_dummy, y)
logis.classes_
logis.coef_
logis.intercept_
logis.score(X_dummy,y)
# 默认参数下精度
# 0.8316498316498316
# 进行超参数搜索
param_grid = {'C':list(np.linspace(0.5, 2, 11))}
logis_grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=5, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.3, random_state=29)
logis_grid.fit(X_train, y_train)
logis_grid.score(X_train, y_train)
# 0.826645264847512
logis_grid.best_score_
logis_grid.best_params_
logis_grid.best_estimator_
type(logis_grid.best_estimator_)
# 测试集上的性能
logis_grid.score(X_test, y_test)
# 预测
y_pred = logis_grid.predict(test_dummy)
y_pred_df = pd.DataFrame(y_pred, index=test_dummy.index)
y_pred_df.columns = ['Survived']
y_pred_df.to_csv("logistic_pred.csv")