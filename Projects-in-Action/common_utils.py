import numpy as np
import pandas as pd

def missing_summary(df: pd.DataFrame, only_miss: bool = True):
    """
    用于统计DF中各列的缺失值信息和占比。
    only_miss: 是否返回所有的列，False只返回有缺失值的列
    """
    # df.isnull()，此方法是 DataFrame.isna 的别名，返回一个和原始 df shape 一样的DataFrame，元素为bool类型，True表示对应位置为缺失值
    miss_num = df.isnull().sum()   # 等价于sum(axis=0)
    if only_miss:  # 只展示有缺失值的特征
        miss_num = miss_num[miss_num > 0]
    # 总样本量
    total = df.shape[0]
    # 缺失值占比（%）
    miss_percent = miss_num / total * 100
    # 各个特征的取值
    cols_values = pd.Series({col: df[col].unique() for col in df})
    # 汇总信息
    miss_info = pd.concat(
        objs=[miss_num, miss_percent, df.dtypes, cols_values],
        axis=1, keys=['miss_num', 'miss_percent', 'dtype', 'values']
    )
    # 统计各个特征的取值个数
    miss_info['values_cnt'] = miss_info['values'].apply(len)
    miss_info['total'] = total
    miss_info = miss_info[['miss_num', 'total', 'miss_percent', 'dtype', 'values_cnt', 'values']]
    miss_info.dropna(inplace=True)
    # 降序排列，同时设置小数点的位数
    miss_info = miss_info.sort_values(by=['miss_num'], ascending=False).round(decimals=3)
    return miss_info


if __name__ == '__main__':
    ...
