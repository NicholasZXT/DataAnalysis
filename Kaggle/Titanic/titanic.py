import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


#查看数据
# %cd Kaggle/Titanic/
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# describe只会给出数值型变量的统计信息，没有多少用处
# train.describe()
# info()可以看出，age, cabin, embarked 含有缺失值
train.info()
test.info()

# 将乘客ID设为index，这个变量对于预测没有用
# 但是
train.set_index('PassengerId', inplace=True)
test.set_index('PassengerId', inplace=True)

# 查看缺失值个数以及占比
train.isnull().sum().sort_values(ascending=False)
(train.isnull().sum()/train.isnull().count()*100).sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
(test.isnull().sum()/test.isnull().count()*100).sort_values(ascending=False)

train['Embarked'].value_counts()
train['Embarked'].mode()[0]

# 选择用于预测的变量
# 这一步不要做的太早，要检查完缺失值之后再做，并且也不要根据自己的猜测随意删减变量
# cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = train[selected_features]
y = train['Survived']
X_test = test[selected_features]


# 填充缺失值，有两种方式
# 1. 使用imputer处理会报错，因为含有 非数值的列sex
# imp_mean = SimpleImputer(strategy='mean')
# X_imp = imp_mean.fit_transform(X)
# 2. 更好的方式是使用pandas的缺失值填充方法
age_fillna_train = X['Age'].mean()
X['Age'].fillna(value=age_fillna_train, inplace=True)
# 注意，对于测试数据也要处理缺失值
age_fillna_test = X_test['Age'].mean()
X_test['Age'].fillna(value=age_fillna_test, inplace=True)
# 测试集的Fare还有一个缺失值
fare_fillna_test = X_test['Fare'].mean()
X_test['Fare'].fillna(value=fare_fillna_test, inplace=True)
# 检查是否还有缺失值
# X.info()
# X_test.info()

# 因为Sex是一个离散变量，需要进行特征处理，这里有三种方式
# 1. 使用sklearn.processing的OneHotEncoder
# 默认会将所有的列都当做离散变量处理，不太好用
# enc = OneHotEncoder(sparse=False)
# t = enc.fit_transform(X)
# enc.get_feature_names()
# 2. 使用pandas的get_dummies()函数更容易处理
X_proc = pd.get_dummies(X)
X_test_proc = pd.get_dummies(X_test)
# 3. 使用DictVectorizer构造特征
# from sklearn.feature_extraction import DictVectorizer
# dict_vec = DictVectorizer(sparse=False)
# X_proc = dict_vec.fit_transform(X.to_dict(orient='record'))
# X_test_proc = dict_vec.fit_transform(X_test.to_dict(orient='record'))
# dict_vec.get_feature_names()
# X_df = pd.DataFrame(X_proc, columns=dict_vec.get_feature_names())

# 处理离散特征时，上面三种方式都可以使用，但是我偏向于使用get_dummies，
# 因为另外两种得到的是np.array，没有列名，而使用get_dummies得到的仍然是DF





# -----------------开始训练模型----------------------------
# Logistic
#预测正确率为 0.74162
logic = LogisticRegression()
logic.fit(X_proc, y)
logic.classes_
logic.coef_
logic.intercept_
logic.score(X_proc, y)
y_test = logic.predict(X_test_proc)
y_test_df = pd.DataFrame(y_test, index= X_test.index)
y_test_df.columns = ['Survived']
y_test_df.to_csv("logis_pred.csv")

# 使用交叉验证的Logistic,
log_cv = LogisticRegressionCV(Cs=np.linspace(0.5,1.5,5), cv=5, max_iter=1000)
log_cv.fit(X_proc, y)
log_cv.coef_
y_test = log_cv.predict(X_test_proc)
y_test_df = pd.DataFrame(y_test, index=X_test_proc.index)

# 使用Random Forest
rfc = RandomForestClassifier(n_estimators=30, random_state=2)
rfc.fit(X_proc, y)
rfc.n_estimators
rfc.n_features_
# 看一下泛化能力
cross_val_score(rfc, X_proc, y, cv=5)
cross_val_score(rfc, X_proc, y, cv=5).mean()
# 预测
y_test = rfc.predict(X_test_proc)
y_test_df = pd.DataFrame(y_test, index=X_test.index)


# 使用XGBoost
from  xgboost import XGBClassifier
# 使用默认参数配置
xgbc = XGBClassifier()
xgbc.fit(X_proc,y)
xgbc.get_booster()
xgbc.get_params()
# 使用交叉验证评估一下效果
cross_val_score(xgbc, X_proc, y, cv=5)
# 预测
y_test = xgbc.predict(X_test_proc)
y_test_df = pd.DataFrame(y_test, index=X_test.index)



# ---------------------------------------
# TODO kaggle实战里的代码部分
dtype = {'PassengerId': str}
train_all = pd.read_csv("train.csv", dtype=dtype)
# 根据列索引来删除某一列
train = train_all.drop(train_all.columns[1], axis=1)
# train['PassengerId'] = train['PassengerId'].astype(str)
test = pd.read_csv("test.csv", dtype=dtype)
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]
# 采用下面这种方式得到的是df，不过训练的时候使用这个df会报warning，
# y_train = train_all.iloc[:, [1]]
# 还是老老实实的使用获取的Series
y_train = train_all['Survived']

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