import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------- numpy 和 pandas 练习 ----------------------
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


planets = sns.load_dataset('planets')

df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'], 'data': range(6)}, columns=['key', 'data'])

for method, group in planets.groupby("method"):
    print(method, group.shape)

t = planets.groupby("method")['year'].describe()

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'], 'data1': range(6), 'data2': rng.randint(0, 10, 6)},
                  columns=['key', 'data1', 'data2'])
df.groupby("key").aggregate(['min', np.median, 'max'])


# ----------------numpy ravel-------------------------------------\
arr = np.arange(0,6).reshape((2,3))
a1 = arr.ravel()
a2 = arr.ravel("F")




# ---------numpy 随机数----------------
np.random.rand(2,3)

np.random.randn(2,3)

np.random.choice(5,3)
np.random.choice([1,3,5,7,9], 3)


# ------------------------matplotlib绘图---------------------------------
# 查看可用的图形风格
plt.style.available
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


# ----------------------------seaborn绘图------------------------------------
plt.style.use("ggplot")

titanic = sns.load_dataset('titanic')
fmri = sns.load_dataset('fmri')


titanic.groupby(['sex', 'class'])['survived'].aggregate('mean')
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
titanic.pivot_table(index='sex', columns='class', values='survived')

titanic.pivot


df = pd.DataFrame({'key': ['2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15'], 'value': [2, 3, 4, 5]})
fig = sns.barplot(data=df, x='key', y='value')
fig.set_xticklabels(fig.get_xticklabels(),rotation=30)
# plt.bar(df['key'],height=df['value'])
plt.show()


# --------------sklearn---------------

import sys

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








if __name__ == "__main__":
    args = sys.argv
    print("name:", args[0])
    print(args[1])
    print(args)
