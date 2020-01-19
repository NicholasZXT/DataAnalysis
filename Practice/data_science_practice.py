import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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