import os
# import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row

if __name__ == '__main__':
    spark = SparkSession.builder.master('local[*]').appName('HelloSpark').getOrCreate()
    # file = r'D:\Projects\DataAnalysis\test-1.csv'
    # df = spark.read.text(file)
    # pd_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    records = [Row(a=1, b=2, c=3), Row(a=4, b=5, c=6)]
    df = spark.createDataFrame(records)
    df.show()
    # df_all = df.collect()
    df.write.csv('hellospark.csv', header=True)
    df.write.text('hellospark.csv')
    df.write.insertInto()
    df.count()
