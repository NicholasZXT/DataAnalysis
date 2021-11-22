import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


# ====================== numpy 练习 =========================================
def __Numpy_Practice():
    pass

# ---------numpy 随机数----------------
rng = np.random.RandomState(0)

# 测试数组占据的内存大小
a1 = rng.rand(1000, 1000)
a2 = rng.rand(10000, 1000)
a3 = rng.rand(10000, 10000)
a4 = rng.rand(20000, 10000)
a5 = rng.rand(20000, 20000)
a11 = a1.astype(np.float32)
# __sizeof__() 返回的单位是 Byte, 转成 MB
a1.__sizeof__()/(1024*1024)   # 7.63 MB, 100 0000 个 float 数
a11.__sizeof__()/(1024*1024)
a2.__sizeof__()/(1024*1024)   # 76.3 MB, 1000 0000 个 float 数
a3.__sizeof__()/(1024*1024)   # 762.94 MB, 1 0000 0000 个
a4.__sizeof__()/(1024*1024)   # 1525.88 MB, 2 0000 0000 个
a5.__sizeof__()/(1024*1024)   # 3051.76 MB, 4 0000 0000 个
# 另外 ndarray.nbytes 属性里存储了数据使用空间大小，它和 __sizeof__() 返回的大小之差是其他元属性的空间大小，通常为 112 Byte
a1.nbytes
a2.nbytes
# 这里使用的是 64 位的 float，一个float数占用 8 个 Byte
# 以 a1 为例，它有 100 0000 个数，占用为 800 0000 Byte，刚好是 a.nbyte 返回的大小

aa = rng.rand(159, 14400)
aa.__sizeof__()/(1024*1024)
aa.__sizeof__()/(1024*1024)*390/150
aa.__sizeof__()/(1024*1024)*390/150*30
aa.__sizeof__()/(1024*1024)*390/150*30/159
aa = rng.rand(1, 96*390)
aa.__sizeof__()/(1024*1024)

np.random.rand(2,3)
np.random.choice(5,3)
np.random.choice([1,3,5,7,9], 3)

t = np.arange(10)
np.roll(t, -1)


# ----------------numpy ravel-------------------------------------\
arr = np.arange(0,6).reshape((2,3))
a1 = arr.ravel()
a2 = arr.ravel("F")


# 掩码操作
a = np.array([1,2,3,4,5])
a_mask = np.ma.masked_array(a, [0,0,0,1,0])

# 求分位数
a = np.arange(0, 10)
np.percentile(a, q=50)
b = [0,0,0,0,0,6,7,8,9]
np.percentile(b, q=50)

a = np.array([[10, 7, 4], [3, 2, 1]])
# 求所有元素的分位数
np.percentile(a, 50)
# 求每一列的分位数
np.percentile(a, 50, axis=0)
# 保留数组的维度
np.percentile(a, 50, axis=0, keepdims=True)
# 求每一行的分位数
np.percentile(a, 50, axis=1)
np.percentile(a, 50, axis=1, keepdims=True)

# 分位数的方法对于掩码矩阵没有效果
a = np.array([[4, 7, 9, 10], [1, 2, 3, 4]])
a_mask = np.ma.masked_array(a, mask=a==2)
np.percentile(a_mask, 50, axis=1, keepdims=True)
# 得使用其他的方式
a = np.array([[4, 7, 9, 10], [4, 7, np.nan, 10], [1, 2, 3, 4], [1, np.nan, 3, 4]])
np.percentile(a, 50, axis=1, keepdims=True)
np.nanpercentile(a, 50, axis=1, keepdims=True)


def fun(a):
    a[np.isnan(a)] = 123
    a[1,1] = 123

a = np.array([[True, True, True], [True, False, True]])

# numpy 插值 + 差分
a = np.arange(12).reshape(3,4)
a = a.astype(np.float32)
a[0,0] = np.nan
a[1,1] = np.nan
a[2,2] = np.nan
a[0,3] = np.nan

a_diff = np.diff(a, axis=1)
a_diff_nan = np.isnan(a_diff)
a1 = pd.DataFrame(a).interpolate(method='linear', limit_area='inside', axis=1).values
a1_diff = np.diff(a1, axis=1)
a1_diff_nan = np.isnan(a1_diff)
a1_diff_not_nan = ~np.isnan(a1_diff)

a_diff_nan & a1_diff_nan
a_diff_nan & a1_diff_not_nan

a_miss = a_diff_nan & a1_diff_not_nan

a = np.array([np.nan, 1.0, 2.0, np.nan, 4.0, 5.0, np.nan]).reshape(1, -1)
a1 = pd.DataFrame(a).interpolate(method='linear', limit_area='inside', axis=1).values
a1_diff = np.diff(a1, axis=1)
a_diff = np.diff(a, axis=1)


# ========================= pandas 练习 ==================================================
def __Pandas_Practice():
    pass

arr = np.arange(0,6).reshape((2,3))
df = pd.DataFrame(arr, index=['r1','r2'], columns=['c1','c2','c3'])

arr
df
arr.sum(axis=0)
df.sum(axis=0)
arr.sum(axis=1)
df.sum(axis=1)

df
df.sum(axis=0)
df.apply(np.sum, axis=0)


df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'], 'data': range(6)}, columns=['key', 'data'])

planets = sns.load_dataset('planets')
for method, group in planets.groupby("method"):
    print(method, group.shape)

t = planets.groupby("method")['year'].describe()

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'], 'data1': range(6), 'data2': rng.randint(0, 10, 6)},
                  columns=['key', 'data1', 'data2'])
df.groupby("key").aggregate(['min', np.median, 'max'])

t = df.to_dict(orient='records')

arr = np.arange(8).reshape((4, 2))
df = pd.DataFrame(arr, columns=['c1', 'c2'])

t = df.rolling(window=2)

for d in t:
    print(type(d))
    print(d)
    print(d.sum())

df
df.shift(periods=-1) - df
df.rolling(window=2).apply(lambda df_temp: df_temp.sum())
df.rolling(window=2, on='c1').apply(lambda df_temp: df_temp.sum())
df.rolling(window=2, on='c1').apply(lambda df_temp: df_temp.iloc[1, :] - df_temp[0, :])

df = pd.DataFrame([(0.0, np.nan, -1.0, 1.0), (np.nan, 2.0, np.nan, np.nan), (2.0, 3.0, np.nan, 9.0), (np.nan, 4.0, -4.0, 16.0)], columns=list('abcd'))
df.interpolate(method='linear', axis=0, limit_area='inside')


# ------------------------------- 时间序列相关 ------------------------------------
def __Pandas_TimeSeries():
    pass

# 构建一个DF
dates = ['2021-08-01', '2021-08-02', '2021-08-03', '2021-08-04', '2021-08-05']
no = np.arange(1, 6)
df = pd.DataFrame({'no': no, 'date': dates})
# 查看数据类型
df.info()
# 转换成 datetime 类型
df['datetime'] = df['date'].apply(lambda date: datetime.strptime(date, '%Y-%m-%d'))
# 可以看出 datetime 这一列本来是 datetime类型，但是pandas 转成了 datetime64[ns] 类型
df.info()
# 根据日期过滤，可以使用 Series 的 .dt 属性——它存储了时间序列类型
t1 = df[df['datetime'].dt.day == 3]


# ============================== matplotlib 练习 ====================================
def __Matplotlib_Practice():
    pass

# 查看可用的图形风格
print(plt.style.available)
# 使用图形风格
plt.style.use("ggplot")

# 查看是否激活绘图
plt.isinteractive()

# 饼图
x = [2,3,4,5]
label = ['a','b','c','d']
plt.pie(x, labels=label)
plt.show()

# 条形图
label = ['b','a','c','d']
plt.bar(label,x)


# =================================== seaborn绘图 ==============================================
def __Seaborn_Plot():
    pass

plt.style.use("ggplot")

titanic = sns.load_dataset('titanic')
fmri = sns.load_dataset('fmri')
planets = sns.load_dataset('planets')

titanic.groupby(['sex', 'class'])['survived'].aggregate('mean')
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
titanic.pivot_table(index='sex', columns='class', values='survived')

titanic.pivot


df = pd.DataFrame({'key': ['2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15'], 'value': [2, 3, 4, 5]})
fig = sns.barplot(data=df, x='key', y='value')
fig.set_xticklabels(fig.get_xticklabels(),rotation=30)
# plt.bar(df['key'],height=df['value'])
plt.show()


# ============================== sklearn ==========================================
def __Sklearn_Practice():
    pass


from sklearn.impute import SimpleImputer
df = pd.DataFrame({'a':[1,2,2,np.nan], 'b':['A','A',None, 'B']})

df.fillna(0)
df.fillna(method='ffill')
df.fillna(method='backfill')

imp = SimpleImputer(strategy='mean')
imp.fit(df)
imp = SimpleImputer(strategy='most_frequent')
imp.fit(df)
imp = SimpleImputer(strategy='constant', fill_value='FILL')
imp.fit(df)
imp.transform(df)


def __Main_location():
    pass


if __name__ == "__main__":
    args = sys.argv
    print("name:", args[0])
    print(args[1])
    print(args)
