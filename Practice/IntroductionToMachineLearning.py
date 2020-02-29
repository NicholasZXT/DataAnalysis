# 科学计算常用的package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 数据集
from sklearn import datasets
# 模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# 模型评估
from sklearn.model_selection import KFold
X = ["a", "a", "b", "c", "c", "c"]
kf = KFold(n_splits=4)
kf.get_n_splits()
kf.split(X)
list(kf.split(X))
for train_indices, test_indices in kf.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])

list(kf.split(X,y))
list(kf.split(X))


from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(estimator=svc, param_grid=parameters, n_jobs=-1)
type(clf)
clf.fit(iris.data, iris.target)
clf.best_score_
clf.best_estimator_
clf.best_estimator_.C
clf.best_params_
clf.cv
clf.cv_results_
clf.get_params()

import numpy as np
from sklearn.model_selection import train_test_split
X = np.arange(10).reshape((5, 2))
y = list(range(5))
X
y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)
X_train
y_train
X_test
y_test


from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
v.fit(D)
v.vocabulary_
v.feature_names_
X = v.transform(D)
type(X)
X
v.inverse_transform(X)
v.get_feature_names()

v_sparse = DictVectorizer()
v_sparse.fit(D)
v_sparse.transform(D)

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = CountVectorizer()

vectorizer.fit(corpus)

vectorizer.vocabulary_
vectorizer.stop_words_

vectorizer.get_stop_words()
vectorizer.get_feature_names()

X = vectorizer.transform(corpus)
type(X)
X.toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

vectorizer.vocabulary_
vectorizer.stop_words_

vectorizer.get_stop_words()
vectorizer.get_feature_names()

X = vectorizer.transform(corpus)
type(X)
X.shape
X.toarray()

import numpy as np
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_data = [[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]
imp_mean.fit(imp_data)
imp_mean.statistics_
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
imp_mean.get_params()
imp_mean.transform(X)

from sklearn.preprocessing import Binarizer
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
transformer = Binarizer()
type(transformer)
transformer.fit(X)

transformer.transform(X)

from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
scaler.fit(data)
scaler.data_max_
scaler.min_
scaler.scale_
scaler.data_min_
scaler.data_max_
scaler.data_range_
scaler.transform(data)

from sklearn.preprocessing import Normalizer
X = [[4, 1, 2, 2],
     [1, 3, 9, 3],
     [5, 7, 5, 1]]
X = np.array(X)
X_norm = Normalizer().fit_transform(X)
X_norm
X_norm[0, :]**2
(X_norm[0, :]**2).sum()
((X_norm[0, :]**2).sum())**(1/2)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)
enc.categories_
enc.transform([['Female', 1], ['Male', 4]]).toarray()
X
enc.transform(X).toarray()
enc.get_feature_names()

from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
accuracy_score(y_true, y_pred,normalize=False)


import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
fpr
tpr
thresholds

import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
pred_prob = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred_prob, pos_label=2)
metrics.auc(fpr, tpr)


from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))