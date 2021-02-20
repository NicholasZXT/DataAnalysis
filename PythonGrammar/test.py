import sys
import os
from elasticsearch import Elasticsearch
import configparser
import numpy as np
import pandas as pd


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



# ------------------数据库操作----------------
# pymysql
import pymysql
connection = pymysql.connect(host='localhost', user='root', password='mysql2020', port=3306, database='crashcourse')
cursor = connection.cursor()
query = "select * from products"
cursor.execute(query=query)
row_1 = cursor.fetchone()
all_rows = cursor.fetchall()
cursor.close()
connection.close()

# MySQL-Client
import MySQLdb
connection = MySQLdb.connect(host='localhost', user='root', passwd='mysql2020', port=3306, db='crashcourse')
cursor = connection.cursor()
query = "select * from products"
cursor.execute(query=query)
row_1 = cursor.fetchone()
all_rows = cursor.fetchall()
cursor.close()
connection.close()


d = {
'a' : [1, 2, 3],
'b' : [4, 5]
}

from collections import  defaultdict

d = defaultdict(lambda :{'args':1, 'keargs': {}})

d['a']


d = {'a':1, 'b':2, 'c':3}

a, b, c = d['a'], d['b'], d['c']


s = set(['a', 'b'])


from datetime import datetime, timedelta

t1 = datetime.strptime('2021-01-28 10:11:21', '%Y-%m-%d %H:%M:%S')
t2 = datetime.strptime('2021-01-29 08:00:11', '%Y-%m-%d %H:%M:%S')
t3 = datetime.strptime('2021', '%Y')

t1.hour
t2.hour

t4 = t2 - t1
t4.days
t4.seconds
t4.max

df = pd.DataFrame({'col1':list("aabbc"), 'col2':list("AABBC"), 'col3':[1,2,3,4,5]})
df.groupby(['col1','col2'], as_index=False).apply(lambda x: x.iloc[0,0])

df.groupby(['col1','col2'], as_index=False).apply(lambda x: x.sort_values(['col3'], ascending=False).nlargest(1, 'col3'))


df[['col1', 'col2']].set_index('col1')['col2'].to_dict()

df['col4'] = df.apply(lambda x: list("abc"), axis=1)

for (k,v) in df.groupby(['col1','col2'], as_index=False):
    print(k)
    print(v)

import getopt


if __name__ == "__main__":
    args = sys.argv
    print("name:", args[0])
    print(args)

    # t = os.environ.get("es", "localhost:19200")

    # 测试getopt函数
    opts, pargs = getopt.getopt(sys.argv[1:], "n:m:", ['param1=', 'param2='])
    print("opts: ", opts)
    print("pargs: ", pargs)