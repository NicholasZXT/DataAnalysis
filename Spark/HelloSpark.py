import os
import pandas as pd
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row


def foreach_partion_fun(index, iter):
    res = []
    for item in iter:
        res.append("[partition index: {}, contents: {}]".format(index, item))
    return res


def foreach_partion_count(partition):
    count = 0
    for item in partition:
        count += 1
        # 下面这个 if 只是为了保证打印一次，不需要每次都打印
        if count == 1:
            # 这里可以看出，DataFrame.rdd.mapPartitions 传进来的是 RDD[Row] 的对象
            print('item.__class__: ', item.__class__)
            print('item: ', item)
    print('partition.__class__: ', partition.__class__, '; partition.count: ', count)
    # 返回值必须是可迭代的类型
    return [count]


if __name__ == '__main__':
    # data_dir = r'D:\Desktop\关口项目\冀北一体化关口数据验证\湖南中台验证'
    data_dir = r'D:\Project-Workspace\Python-Projects\DataAnalysis\datasets\zshield'
    conf = SparkConf().setAppName('HelloSpark').setMaster('local[2]')
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # file = r'D:\Projects\DataAnalysis\test-1.csv'
    # df = spark.read.text(file)
    # records = [Row(a=1, b=2, c=3), Row(a=4, b=5, c=6)]
    # df = spark.createDataFrame(records)
    # df.show()
    # df_all = df.collect()
    # df.write.csv('hellospark.csv', header=True)
    # df.write.text('hellospark.csv')

    # 由 DataFrame 转成的 RDD，返回的是 RDD[Row] 对象
    rng = np.random.RandomState(0)
    key_col = list('ABCAABBBCCCC')
    col_2 = range(len(key_col))
    col_3 = rng.randint(0, 10, len(key_col))
    df_pd = pd.DataFrame({'key': key_col, 'data1': col_2, 'data2': col_3})
    df = spark.createDataFrame(df_pd)
    # df.show()
    df_rdd = df.rdd
    # df_rdd.foreach(print)  #显示的是Row对象
    # df.foreachPartition(lambda par: print("partition.__class__: {}".format(par.__class__)))
    # 重新分区
    # df_rep = df.repartition('key')  # 如果不指定分区数，就默认是 200，产生 200 个Task
    df_rep = df.repartition(10, 'key')
    print("partition num: ", df_rep.rdd.getNumPartitions())
    # df_rep.foreachPartition(foreach_partion_count)
    df_rep_par = df_rep.rdd.mapPartitions(foreach_partion_count)
    print(df_rep_par.__class__)
    df_rep_par.foreach(print)

    # equip_subs_path = os.path.join(data_dir, 'LOSS_ARCH_EQUIP_SUBS-test.csv')
    # equip_subs_rdd = sc.textFile(equip_subs_path, minPartitions=3)
    # rdd_2 = equip_subs_rdd.mapPartitionsWithIndex(foreach_partion_fun)
    # print(rdd_2.take(5))