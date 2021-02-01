import sys
import os
from elasticsearch import Elasticsearch
import configparser


es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# es = Elasticsearch([{'host': 'localhost', 'port': 9200}], http_auth=('xiao', '123456'), timeout=3600)

result = es.indices.create(index='test-index', ignore=400)

result = es.indices.delete(index='test-index')
result = es.indices.delete(index='test')

es.indices.create(index='test', ignore=400)
data = {'title': '美国留给伊拉克的是个烂摊子吗', 'url': 'http://view.news.qq.com/zt2011/usa_iraq/index.htm'}
result = es.create(index='test', doc_type='_doc', id=1, body=data)

es.cat.indices()


# 获取路径中的最后一个文件夹名称
os.path.basename('/path/to/Kaggle')
# 获取文件夹路径
os.path.dirname('/path/to/Kaggle')


config = configparser.ConfigParser()
config['DEFAULT'] = {'ServerAliveInterval': '45',
                     'Compression': 'yes',
                     'CompressionLevel': '9'}
config['bitbucket.org'] = {}
config['bitbucket.org']['User'] = 'hg'
config['topsecret.server.com'] = {}
topsecret = config['topsecret.server.com']
topsecret['Port'] = '50022'     # mutates the parser
topsecret['ForwardX11'] = 'no'  # same here
config['DEFAULT']['ForwardX11'] = 'yes'
with open('config.ini', 'w') as configfile:
  config.write(configfile)


config.sections()
config.get(section='bitbucket.org', option='User')
config.get(section='bitbucket.org', option='nothing', fallback='not exist')



if __name__ == "__main__":
    args = sys.argv
    print("name:", args[0])
    print(args)

    t = os.environ.get("es", "localhost:19200")