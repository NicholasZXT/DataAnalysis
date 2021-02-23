import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. 'startup': for $1 billion")
doc.has_vector
doc.vector.__class__
doc.vector.shape
doc.vector_norm.__class__
doc.vector_norm
doc.tensor.__class__
doc.tensor.shape
doc.tensor # 句子中所有word的 词向量组成的矩阵，每一行对应于一个词的向量
len(doc)
doc[8] # 这个是美元符号
doc[9]
doc[8].vector
doc[6]
doc[6].vector # 标点符号和美元符号也有词向量

for token in doc:
    print(token.text, token.pos_, token.dep_)

# 模型的词典长度
len(nlp.vocab)
t1 = doc[0]
t1.text
t1.has_vector
t1.i
t1.idx
t1.lemma
t1.lemma_
t1.is_stop
t1.is_alpha
t1.is_punct

t1_vec = t1.vector
t1_vec.__class__
t1_vec.shape

doc[4]
doc[5]

## ----------------------------IMDB数据集---------------------------------------
from sklearn.datasets import load_files

imdb_train = load_files('./datasets/aclImdb/train/')
imdb_test = load_files('./datasets/aclImdb/test/')

text_train, label_train = imdb_train.data, imdb_train.target
text_train.__class__
len(text_train)
label_train.__class__
label_train.shape
text_train[0].__class__  #其中的字符串是bytes类型
text_train[0]
label_train[0]
t = text_train[0]
t = b"1"

# 转换成 str 类型，并去除掉一些html符号
text_train = [doc.decode("utf-8").replace("<br />", " ") for doc in text_train]
text_train[0].__class__
t = text_train[0]
# id(text_train[0])
# id(t)

def sentence2vector(doc):
    """
    用于从 sentence 中过滤出有用的word
    @param sentence: spacy的doc对象，封装了一个句子
    @return: 句子中有效单词所组成的 词向量矩阵,
    """
    doc_matrix = doc.tensor
    # 接下来要从上面剔除掉 标点符号 等无效部分的词向量
    index_list = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_currency:
            continue
        else:
            index_list.append(token.i)
    return doc_matrix[index_list, :]


doc = nlp(text_train[0])
doc = nlp(text_train[1])
doc_matrix = sentence2vector(doc)
doc.tensor.shape
doc_matrix.shape

doc = nlp("html <br /> symbor")
for token in doc:
    print(token)

doc[1]
doc[1].doc
doc[1].tensor.shape
doc[1].vector

##-----numpy数组拼接-------

a1 = np.ones(shape=(2, 2))
a2 = np.ones(shape=(2, 2))*2
a3 = np.ones(shape=(2, 2))*3
a = np.concatenate([a1, a2, a3])
a = np.array([a1, a2, a3])
a.shape


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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


# -----------------stanfordSentimentTreebank数据集-------------------------------




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