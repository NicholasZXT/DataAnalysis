import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for token in doc:
    print(token.text, token.pos_, token.dep_)

# 模型的词典长度
len(nlp.vocab)
t1 = doc[0]
t1_vec = t1.vector
t1_vec.__class__
t1_vec.shape

from sklearn.datasets import load_files

imdb_train = load_files('./datasets/aclImdb/train/')
imdb_test = load_files('./datasets/aclImdb/test/')

text_train, label_train = imdb_train.data, imdb_train.target
text_train.__class__
len(text_train)
label_train.__class__
label_train.shape
text_train[0]
label_train[0]

text_train = [doc.decode("utf-8").replace("<br />", " ") for doc in text_train]
t = text_train[0]
t.__class__
t.decode('utf-8')

from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer
count_vec = CountVectorizer()
count_vec.fit(text_train)
len(count_vec.vocabulary_)
count_vec.vocabulary_.__class__
text_train_cnt = count_vec.transform(text_train)
text_train_cnt.shape

tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(text_train)
text_train_tfidf = tfidf_vec.transform(text_train)
text_train_tfidf.shape


# 读取 stanfordSentimentTreebank 数据集




# 将所有的文本都转换成词向量

doc = nlp(text_train[0])
doc[0]

def text2vec(doc):
    """
    使用 spacy 将文本转成词向量
    @param doc: list of str
    @return:
    """
    nlp = spacy.load("en_core_web_sm")