import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

logreg = LogisticRegression()
cross_val_score(logreg, X, y, cv=5)

logreg.fit(X_train, y_train)
y_pred = logreg.predict_proba(X_test)[:, 1]

fpr, tpr, thresh = roc_curve(y_test, y_pred)


#  缺失值
X = np.arange(12).reshape((3, 4)).astype(float)
X
X[[0,1,2],[0,1,2]]
Y = np.arange(12,24).reshape((3,4)).astype(float)
Y
Y[[0,1,2],[0,1,2]] = np.NaN
Y
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit(X)
imp.transform(Y)


