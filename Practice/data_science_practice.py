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

titanic = sns.load_dataset('titanic')
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean')
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
titanic.pivot_table(index='sex', columns='class', values='survived')

titanic.pivot

fmri = sns.load_dataset('fmri')

# ---测试seaborn绘图
plt.style.use("ggplot")
df = pd.DataFrame({'key': ['2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15'], 'value': [2, 3, 4, 5]})
fig = sns.barplot(data=df, x='key', y='value')
fig.set_xticklabels(fig.get_xticklabels(),rotation=30)
# plt.bar(df['key'],height=df['value'])
plt.show()


# ---------numpy 随机数----------------
np.random.rand(2,3)

np.random.randn(2,3)

np.random.choice(5,3)
np.random.choice([1,3,5,7,9], 3)
# ------------------------Python绘图---------------------------------
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


# --------------seaborn-------------
x = np.random.randn(2,10)
sns.scatterplot(x[0,:], x[1,:])