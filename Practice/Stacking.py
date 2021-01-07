import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor




class StackingRegressor(BaseEstimator, RegressorMixin):
    """
    这里尝试构建一个Stacking的类，完成的功能是，给定第一层的基础估计器（以字典存储），完成第一层的基础分类器的训练，并保存训练过程中的模型，
    一共需要训练的模型个数为 n x cv，n是基础估计的个数，cv是交叉划分的折数。
    transform方法用于对训练集test_X做变换（不是对X_train做变换）
    --但是这个尝试失败了，原因是，第一层基础分类器构建时，与对训练集train_X的K折交叉划分是紧密相关的，虽然可以保存所有训练的模型，但是由于要对原始
    的训练集train_X做变换，输出这个变换的结果给第二层分类器使用，这里不太好处理
    """

    def __init__(self, base_model_dict, second_model, cv=5):
        super().__init__()
        # 交叉验证的折数
        self.cv = cv
        # 第一层的base model中，每个模型都要 copy 成交叉验证的 cv 份
        self.base_model_dict = {model_name: [clone(model) for i in range(self.cv)] for model_name, model in
                                base_model_dict.items()}
        # 第二层模型
        self.second_model = second_model
        # 用于存放第一层模型在K-折训练集上的 out-of-fold 预测结果，它的列数=第一层模型个数，行数=训练数据的长度（暂时空着,直到fit方法里才会被设定）
        self.X_oof = np.zeros(shape=(len(base_model_dict),))
        # KFold分割对象
        self.kf = KFold(n_splits=self.cv)

    def fit(self, X, y):
        # 首先根据训练集X的shape设置X_oof的shape（主要是行数）
        self.X_oof = np.zeros(shape=(len(X), len(self.base_model_dict)))
        # 调用一系列base estimator 进行训练，并存储所有模型的 out-of-fold 预测结果
        for col_index, model_name in enumerate(self.base_model_dict.keys()):
            # 对一个base estimator进行KFold训练，并得到训练后的 out-of-fold 预测，得到一列 oof 结果
            train_oof_col = self.get_oof(model_name, X, y)
            self.X_oof[:, col_index] = train_oof_col
        # 得到第一层模型的 out-of-fold 结果后，使用 second_model 进行训练
        self.second_model.fit(self.X_oof, y)

    def get_oof(self, model_name, X, y):
        """
        对给定的模型 model_name，计算 KFold 下每折的 out-of-fold 预测结果并返回
        model_name：str，是 self.base_model_dict的key，它获取的是一个模型list，list长度 = KFold对象的分割次数
        返回值 shape = ( len(X), 1 )
        """
        # 首先根据类的KFold对象，对训练集X进行分割，获取分割索引的生成器
        split = self.kf.split(X)
        # 构建一个数组，用于保存K-折训练集得出的 out-of-fold 预测值,作为返回值，此数组的行数 = 训练集X的行数
        train_oof = np.zeros(shape=(len(X),))
        # 每一折的数据，对应于一个模型
        for split_index, model in zip(split, self.base_model_dict[model_name]):
            # 拆分出每一折的训练集index和测试集index
            train_index, test_index = split_index
            # 获取训练折的数据和测试折的数据
            train_X = X[train_index]
            train_y = y[train_index]
            test_X = X[test_index]  # 测试折的y标签不需要
            # 在训练折数据上训练模型
            model.fit(train_X, train_y)
            # 在测试折的数据上进行预测，并保存下来
            train_oof[test_index] = model.predict(test_X)
        return train_oof

    def predict(self, X):
        """
        给定测试集X，进行预测
        """
        model_num = len(self.base_model_dict)
        model_name_list = self.base_model_dict.keys()
        # X_transform用于存放使用第一层的base model 预测过后的中间结果
        X_transform = np.zeros(shape=(len(X), model_num))
        for model_index, model_name in enumerate(model_name_list):
            # test_oof里存放的是KFold得到的self.cv个模型的预测值，每列一个
            test_oof = np.zeros(shape=(len(X), self.cv))
            for cv_index, model in enumerate(self.base_model_dict[model_name]):
                test_oof[:, cv_index] = model.predict(X)
            # 最终的取值是将self.cv个模型的预测值求平均
            X_transform[:, model_index] = test_oof.mean(axis=1)
        # 使用第二层的模型进行预测
        y_pred = self.second_model.predict(X_transform)
        return y_pred


if __name__ == "__main__":
    kf = KFold()
    kf.get_n_splits()
    X = np.arange(0, 20).reshape((10, 2))
    list(kf.split(X))
    X1 = np.arange(0, 30).reshape((10, 3))
    list(kf.split(X1))

    split = kf.split(X)
    list(enumerate(split))

    step = np.arange(1, 5)
    x, y = np.meshgrid(step, step)
    x_flat = x.ravel()
    y_flat = y.ravel()
    X = np.array(list(zip(x_flat, y_flat)))
    y = X[:, 0] ** 2 + X[:, 1] ** 2

    # 测试StackingRegressor类
    lr = LinearRegression()
    tree = DecisionTreeRegressor()
    rf = RandomForestRegressor()

    model_dict = {'rf': rf, 'tree': tree}
    stacking = StackingRegressor(base_model_dict=model_dict, second_model=lr, cv=3)
    stacking.fit(X, y)
    y_pred = stacking.predict(X)
    stacking.score(X, y)

    stacking.base_model_dict
    X_oof = stacking.X_oof
