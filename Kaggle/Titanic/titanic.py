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
# 采用下面这种方式得到的是df，不过训练的时候使用这个df会报warning，
# y_train = train_all.iloc[:, [1]]
# 还是老老实实的使用获取的Series
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
# 测试集里的Fare还有一个缺失值
fare_fillna_test = X_test['Fare'].mean()
X_test['Fare'].fillna(fare_fillna_test, inplace=True)

# 对特征进行向量化
from sklearn.feature_extraction import DictVectorizer
# DictVectorizer只能处理字典元素的列表
dict_vec = DictVectorizer(sparse=False)
# 需要先将X_train转成dict, orient='record'表示转成list形式的dict
X_train_vec = dict_vec.fit_transform(X_train.to_dict(orient='record'))
dict_vec.feature_names_
X_test_vec = dict_vec.transform(X_test.to_dict(orient='record'))
X_train_vec.shape
X_test_vec.shape
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
# rfc = RandomForestClassifier()
rfc = RandomForestClassifier(n_estimators=25, random_state=2)

# 导入XGBoost
from xgboost import XGBClassifier
xgbc = XGBClassifier()

# 导入交叉验证
from sklearn.model_selection import cross_val_score
# 验证两个模型在默认配置下的效果
cross_val_score(rfc, X_train_vec, y_train, cv=5)
cross_val_score(rfc, X_train_vec, y_train, cv=5).mean()

cross_val_score(xgbc, X_train_vec, y_train, cv=5)
cross_val_score(xgbc, X_train_vec, y_train, cv=5).mean()

# -----------进行预测-------------------
rfc.fit(X_train_vec, y_train)
rfc_y_pred = rfc.predict(X_test_vec)

xgbc.fit(X_train_vec, y_train)
xgbc_y_pred = xgbc.predict(X_test_vec)

# --------采用网格进行超参数搜索-----------------
from sklearn.model_selection import GridSearchCV
params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, param_grid=params, n_jobs=-1, cv=5, verbose=1)
gs.fit(X_train_vec, y_train)
gs.best_score_
gs.best_params_

xbgc_best_pred = gs.predict(X_test_vec)