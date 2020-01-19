import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dtype = {'PassengerId': str}
train_all = pd.read_csv("train.csv", dtype=dtype)
# 根据列索引来删除某一列
train = train_all.drop(train_all.columns[1], axis=1)
# train['PassengerId'] = train['PassengerId'].astype(str)
test = pd.read_csv("test.csv", dtype=dtype)

train_all.shape
train.shape
test.shape

selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train_all.iloc[:, [1]]
X_train.shape
y_train.shape

# 处理缺失值
# 处理 Age
age_fillna_train = X_train['Age'].mean()
X_train['Age'].fillna(age_fillna_train, inplace=True)
age_fillna_test = X_test['Age'].mean()
X_test['Age'].fillna(age_fillna_test, inplace=True)
# 处理embark
X_train['Embarked'].value_counts()
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)

# 对特征进行向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
X_train_vec = dict_vec.fit_transform(X_train.to_dict(orient='record'))
dict_vec.feature_names_
