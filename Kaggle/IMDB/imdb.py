
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV,train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix,classification_report

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re


#加载数据
# %cd Kaggle/IMDB/
# sklearn里的load_files函数可以直接处理IMDB的数据，不用手动处理
# 这里指定了加载的类别pos和neg，避免加载用于无监督学习的unsup
imdb_train = load_files('./aclImdb/train', categories=['neg','pos'])
imdb_test = load_files('./aclImdb/test', categories=['neg','pos'])
text_train = imdb_train.data
y_train = imdb_train.target
text_test = imdb_test.data
y_test = imdb_test.target

# 清洗数据 + 特征工程

# 最简单的词袋表示法
# 1. 默认配置
vect = CountVectorizer()
t1 = text_train[0:5]
t2 = vect.fit_transform(t1)
X_train = vect.fit_transform(text_train)
X_test = vect.transform(text_test)
len(vect.get_feature_names())
# 有74849个特征
# 2. 使用停用词表
vect = CountVectorizer(min_df=5, stop_words='english')
X_train = vect.fit_transform(text_train)
X_test = vect.transform(text_test)
len(vect.get_feature_names())
# 现在只有26967个特征了

# 实际处理
# 文本里面含有html标记，使用bs4去除
t1 = text_train[0]
t2 = BeautifulSoup(t1,'html')
t2.text
#定义处理每条记录的函数
def text_clean(review):
    # 去除HTML标记
    # review = t1
    raw_text = BeautifulSoup(review,'html').get_text()
    # 去除非字母用词
    letters = re.sub("[^a-zA-Z]",' ',raw_text)
    words = letters.lower().split()
    # 去除停用词
    stop_words = set(ENGLISH_STOP_WORDS)
    words = [w for w in words if w not in stop_words]
    # 将上述处理好的word列表拼接起来
    sentence = " ".join(words)
    return sentence

t3 = text_clean(t1)

# 处理特征
text_train = [text_clean(review) for review in text_train]
text_test = [text_clean(review) for review in text_test]


# 训练模型

# 使用LogisticRegression
lr = LogisticRegression(max_iter=1500)
# 单纯的看一下效果
scores = cross_val_score(lr, X_train, y_train, cv=5)
# 使用网格搜索调整正则化参数C
param_grid = {'C':[0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(lr, param_grid, n_jobs=-1, cv=5)
grid.fit(X_train, y_train)
grid.best_score_
grid.best_params_
# 检测最好的网格搜索参数进行预测的结果
grid.score(X_test,y_test)
# 使用Tif以及管道
pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression(max_iter=1500))
param_grid = {'logisticregression__C':[0.001, 0.01, 0.1, 1, 10]}
grid_pipe = GridSearchCV(pipe, param_grid, cv= 5)
# 这里使用的TfidfVectorizer，所以直接传入原有的文本数据就行
grid_pipe.fit(text_train,y_train)
grid_pipe.best_score_
grid_pipe.best_estimator_
type(grid_pipe.best_estimator_.named_steps)
grid_pipe.best_estimator_.named_steps['logisticregression']


# 使用朴素贝叶斯来建模
# 分别使用两组特征,词袋模式和tif
# 1.词袋模型
# 注意，词袋模型在将原本的text_train抓换成数值的X_train_bow时，只是对单个记录进行了转换
# 没有使用整个数据集的信息，所以这里可以在将X_train_bow分割成训练集和测试集之前转换，不会"泄露"信息
count_vec = CountVectorizer(min_df=5)
X_train_bow = count_vec.fit_transform(text_train)
X_test_bow = count_vec.transform(text_test)
# 简单测试下
mnb = MultinomialNB()
mnb.fit(X_train_bow,y_train)
y_pred_count = mnb.predict(X_test_bow)
confusion_matrix(y_test, y_pred_count, labels=[1,0])
mnb.score(X_test_bow,y_test)
# 使用带交叉验证的网格搜索
# 这个参数是mnb的平滑参数
param_count = {'alpha':[0.1, 1.0, 10]}
# 要将样本分为训练集和测试集——这里划分的是经过了处理的X_train_bow，而不是原始文本形式的text_train
train,test,train_y,test_y = train_test_split(X_train_bow,y_train,random_state=29)
# 在训练集上进行网格搜索
grid_bow = GridSearchCV(MultinomialNB(), param_grid=param_count, cv=4)
grid_bow.fit(train,train_y)
grid_bow.best_score_
grid_bow.best_params_
test_y_pred = grid_bow.predict(test)
# 由测试集估计的泛化性能——0.85696
confusion_matrix(test_y, test_y_pred,labels=[1,0])
grid_bow.score(test,test_y)
# 实际的泛化性能——0.83096
grid_bow.score(X_test_bow,y_test)
y_test_pred = grid_bow.predict(X_test_bow)
confusion_matrix(y_test,y_test_pred,labels=[1,0])

# 2.使用tfidf构建特征
# 由于tfidf转换器会使用整个数据集的信息来将text_train转换成数值，
# 所以必须要在这种转换前完成数据集的分割，这里就要用到pipeline
pipe_tfidf = Pipeline([('tfidf_vec',TfidfVectorizer()), ('mnb', MultinomialNB())])
# 构造参数网格，其实TfidfVectorizer也有参数，这里并没有选
# 这里的参数名alpha前就需要带上pipeline里步骤名，并用双下划线分开
param_tfidf = {'mnb__alpha':[0.1, 1.0, 10]}
grid_tfidf = GridSearchCV(pipe_tfidf, param_grid=param_tfidf, cv=4)
# 对原始的文本数据text_train进行划分
train, test, train_y, test_y = train_test_split(text_train,y_train,random_state=29)
grid_tfidf.fit(train, train_y)
# 查看最佳模型信息
grid_tfidf.best_params_
grid_tfidf.best_score_
grid_tfidf.score(train, train_y)
# 检查在训练集上的泛化性能
grid_tfidf.score(test, test_y)
test_y_pred = grid_tfidf.predict(test)
confusion_matrix(test_y, test_y_pred, labels=[1,0])
# 实际的泛化性能
grid_tfidf.score(text_test, y_test)
y_test_pred = grid_tfidf.predict(text_test)
confusion_matrix(y_test, y_test_pred, labels=[1,0])