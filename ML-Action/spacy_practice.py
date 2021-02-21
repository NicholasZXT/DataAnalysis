import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for token in doc:
    print(token.text, token.pos_, token.dep_)

from sklearn.datasets import load_files

imdb_train = load_files('./datasets/aclImdb/train/')
imdb_test = load_files('./datasets/aclImdb/test/')