import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


DATA_DIR = r"D:\Project-Workspace\Python-Projects\DataAnalysis\local-datasets\工业蒸汽量预测"
train_data = pd.read_csv(os.path.join(DATA_DIR, 'zhengqi_train.txt'), delimiter='\t', header=0)
test_data = pd.read_csv(os.path.join(DATA_DIR, 'zhengqi_test.txt'), delimiter='\t', header=0)
X = train_data.drop(columns=['target'])
y = train_data['target'].copy()


