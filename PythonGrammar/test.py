import sys
import os
from elasticsearch import Elasticsearch


es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# es = Elasticsearch([{'host': 'localhost', 'port': 9200}], http_auth=('xiao', '123456'), timeout=3600)

result = es.indices.create(index='test-index', ignore=400)

result = es.indices.delete(index='test-index')
result = es.indices.delete(index='test')

es.indices.create(index='test', ignore=400)
data = {'title': '美国留给伊拉克的是个烂摊子吗', 'url': 'http://view.news.qq.com/zt2011/usa_iraq/index.htm'}
result = es.create(index='test', doc_type='_doc', id=1, body=data)

es.cat.indices()


if __name__ == "__main__":
    args = sys.argv
    print("name:", args[0])
    print(args)

    t = os.environ.get("es", "localhost:19200")