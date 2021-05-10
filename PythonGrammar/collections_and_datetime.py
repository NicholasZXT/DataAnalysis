import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ------------------数据库操作----------------
def sql_practice():
    pass

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


# -------------------------- 日期操作 -----------------------------------
from datetime import datetime, timedelta, timezone
import datetime

def datetime_manipulate():
    """
    日期处理相关的操作
    @return:
    """
    pass

t1 = datetime.strptime('2021-01-28 10:11:21', '%Y-%m-%d %H:%M:%S')
t2 = datetime.strptime('2021-01-29 08:00:11', '%Y-%m-%d %H:%M:%S')
t3 = datetime.strptime('2021', '%Y')
t4 = datetime.strptime('2021-01-29T18:00:11.000Z', "%Y-%m-%dT%H:%M:%S.%f%z")

t4 = datetime.strptime('2021-01-29T09:43:47.000Z', "%Y-%m-%dT%H:%M:%S.%f%z")

datetime.now().strftime("%Y-%m-%d %H:%M:%S")

date_str = '2021-01-29T09:43:47.000Z'
t4 = datetime.strptime(date_str[:-1], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc)

datetime.today().strftime("%Y.%m.%d")
timedelta(days=1)
datetime.today() - timedelta(days=1)

list(range(9, 0, -1))

d = {'a':None, 'b':1}

s = ['Is', 'Chicago', 'Not', 'Chicago?']
'__'.join(s)
"abc".join(['_1', '_2', '_3'])

t1.hour
t2.hour

t4 = t2 - t1
t4.days
t4.seconds
t4.max

df = pd.DataFrame({'col1':list("aabbc"), 'col2':list("AABBC"), 'col3':[1,2,3,4,5]})
df.groupby(['col1','col2'], as_index=False).apply(lambda x: x.iloc[0,0])

df.groupby(['col1','col2'], as_index=False).apply(lambda x: x.sort_values(['col3'], ascending=False).nlargest(1, 'col3'))

# datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
date_list = ["2021-01-10 12:24:21", "2021-01-05 12:12:12", "2021-01-03 08:08:08", "2021-02-01 09:12:13", "2021-01-08 12:12:12",]
df = pd.DataFrame({'col1':list("aabbc"), 'col2':list("AABBC"), 'col3':date_list})
df['col4'] = df['col3'].apply(datetime.strptime, args=("%Y-%m-%d %H:%M:%S", ))
df.groupby(['col1','col2'], as_index=False).apply(lambda x: x.sort_values(['col3'], ascending=False).nlargest(1, 'col3'))
df.groupby(['col1','col2'], as_index=False).apply(lambda x: x.sort_values(['col4'], ascending=False).nlargest(1, 'col4'))


df[['col1', 'col2']].set_index('col1')['col2'].to_dict()

df['col4'] = df.apply(lambda x: list("abc"), axis=1)

for (k,v) in df.groupby(['col1','col2'], as_index=False):
    print(k)
    print(v)

# ------------------ collections 数据结构 --------------------
def collections_practice():
    """
    集合的数据结构相关操作
    @return:
    """
    pass

d = {
'a' : [1, 2, 3],
'b' : [4, 5]
}

from collections import defaultdict

d = defaultdict(lambda :{'args':1, 'keargs': {}})

d['a']


d = {'a':1, 'b':2, 'c':3}

a, b, c = d['a'], d['b'], d['c']

t = (1, 2, 3, [4,5,6])


s = set(['a', 'b'])

from collections import OrderedDict
d = OrderedDict({'a': 1, 'b': 2, 'c': 3})
d.keys()
d.popitem(last=False)
d.popitem(last=False)[0]
d.items()
list(d.keys())

d = {'a': [], 'b': 12}
d['a'].append(2)
d['a'].append(3)

a = ['a', 'b', 'c']
b = ['1', '2', '3']

for i, (a_value, b_value) in enumerate(zip(a, b)):
    print(i, a_value, b_value)


t = [(a_value, b_value) for a_value, b_value in zip(a, b) if (a_value=='a' and b_value=='1')]

d = dict(zip(['a','b','c'], [1,2,3]))
d.items()

for k, v in d.items():
    print(k, ": ", v)

from distutils.version import LooseVersion
t = LooseVersion(np.__version__)