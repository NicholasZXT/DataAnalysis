import os
import pandas as pd
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row

if __name__ == '__main__':
    spark = SparkSession.builder.master('local[*]').appName('HelloSpark').getOrCreate()
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
    df_rdd.foreach(print)  #显示的是Row对象