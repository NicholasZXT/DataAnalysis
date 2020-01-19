# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:59:09 2019

@author: cm
"""
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns

boston = datasets.load_boston()
data = pd.DataFrame(boston.data,columns = boston.feature_names)
target = pd.DataFrame(boston.target,columns = ["MEDV"])
data_concat = pd.concat([data,target],axis = 1)
data_concat_sub = data_concat[["CRIM","RM","AGE","LSTAT","MEDV"]]

sns.pairplot(data_concat[["CRIM","RM","AGE","LSTAT","MEDV"]])

#data_concat.plot.scatter('LSTAT','MEDV')
lstat = data[['LSTAT']]


lm = LinearRegression()
lm.fit(lstat,target)
lm.score(lstat,target)

lstat = sm.add_constant(lstat)
lm = sm.OLS(target,lstat).fit()
lm.summary()
lm.params

lstat_plot = np.linspace(0,40)
plt.scatter(lstat.LSTAT,target)
plt.plot(lstat_plot,lstat_plot*lm.params[1]+lm.params[0],color = 'red')


lm_fitted_y = lm.fittedvalues
lm_residuals = lm.resid
lm_norm_residuals = lm.get_influence().resid_studentized_internal
lm_norm_resid_abs_sqrt = np.sqrt(np.abs(lm_norm_residuals))
lm_abs_resid = np.abs(lm_residuals)
lm_leverage = lm.get_influence().hat_matrix_diag
lm_cooks = lm.get_influence().cooks_distance[0]

sns.residplot(lm_fitted_y,lm.resid,lowess=True)
sm.qqplot(lm_norm_residuals,fit= True)
plt.scatter(lm_fitted_y,lm_norm_resid_abs_sqrt)
sns.regplot(lm_fitted_y,lm_norm_resid_abs_sqrt,lowess = True)
sns.regplot(lm_leverage,lm_norm_residuals,lowess = True)

