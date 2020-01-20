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
y_train = train_all['Survived']

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
# DictVectorizer只能处理字典元素的列表
dict_vec = DictVectorizer(sparse=False)
# 需要先将X_train转成dict, orient='record'表示转成list形式的dict
X_train_vec = dict_vec.fit_transform(X_train.to_dict(orient='record'))
dict_vec.feature_names_
X_test_vec = dict_vec.transform(X_test.to_dict('record'))

# DictVectorizer示例
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Francisco', 'temperature': 18.},
]
df_vec = DictVectorizer(sparse=False)
# 下面这两个会报错
df = pd.DataFrame(measurements)
df_vec.fit_transform(df)
# 这个才是正确的
df_vec.fit_transform(measurements)

# -------开始训练模型--------------
# 导入随机森林
from sklearn.ensemble import RandomForestClassifier
# 这里使用默认的配置会显示warning，提示n_estimators的变化
rfc = RandomForestClassifier()
rfc = RandomForestClassifier(n_estimators=25, random_state=2)

# 导入XGBoost
from xgboost import XGBClassifier
xgbc = XGBClassifier()
# 导入交叉验证
from sklearn.model_selection import cross_val_score

cross_val_score(rfc, X_train_vec, y_train, cv=5)
cross_val_score(rfc, X_train_vec, y_train, cv=5).mean()

cross_val_score(xgbc, X_train_vec, y_train, cv=5)
cross_val_score(xgbc, X_train_vec, y_train, cv=5).mean()