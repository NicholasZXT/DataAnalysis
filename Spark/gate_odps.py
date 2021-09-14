import sys
# print("sys.path: ", sys.path)
import os
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, Row, Column
from pyspark.sql.functions import expr
from pyspark.sql.types import FloatType


def read_data(spark, data_dir=None):
    if data_dir:
        print('reading local data.')
        equip_subs_path = os.path.join(data_dir, 'LOSS_ARCH_EQUIP_SUBS-test.csv')
        model_cell_path = os.path.join(data_dir, 'LOSS_ARCH_MODEL_CELL-test.csv')
        cell_mp_path = os.path.join(data_dir, 'LOSS_ARCH_REL_CELL_MP-test.csv')
        high_path = os.path.join(data_dir, 'HIGH.csv')
        equip_subs_df = spark.read.csv(equip_subs_path, header=True, encoding='utf-8')
        model_cell_df = spark.read.csv(model_cell_path, header=True)
        cell_mp_df = spark.read.csv(cell_mp_path, header=True)
        high_df = spark.read.csv(high_path, header=True)
    else:
        print('reading HFDS data.')
        equip_subs_df = spark.sql('select * from LOSS_ARCH_EQUIP_SUBS')
        model_cell_df = spark.sql('select * from LOSS_ARCH_MODEL_CELL')
        cell_mp_df = spark.sql('select * from LOSS_ARCH_REL_CELL_MP')
        high_df = spark.sql('select * from HIGH')
    return equip_subs_df, model_cell_df, cell_mp_df, high_df

def foreach_subs_test(partition):
    count = 0
    for row in partition:
        count += 1
        if count == 1:
            print("row.__class__: ", row.__class__)
            print("row: ", row)
    print('partition.__class__: ', partition.__class__, '; partition.count: ', count)
    return [count]

def foreach_subs_test_2(partition):
    res_dict = []
    for row in partition:
        res_dict.append(row.asDict())
    res_df = pd.DataFrame(res_dict)
    df_stat = res_df.groupby(['SUBS_ID', 'SUBS_NAME', 'SUBS_ADDR'], as_index=False)['MP_ID'].count()
    df_stat_dict = df_stat.to_dict(orient='records')
    for row in df_stat_dict:
        yield Row(**row)

def foreach_subs_proc(index, partition):
    archive_cols = ['SUBS_NAME', 'SUBS_ID', 'SUBS_TYPE', 'SUBS_ADDR', 'MP_ID', 'MP_NAME', 'GROUP_VOLT_CODE', 'GROUP_ID',
                    'T_FACTOR', 'TF_ID', 'RUN_STATUS']
    data_cols = ['MP_ID', 'DATA_DATE', 'PAP_PHI', 'RAP_PHI']
    res = []
    res_return = []
    for row in partition:
        res.append(row.asDict())
    if len(res):
        df = pd.DataFrame(res)
        print('df.shape: ', df.shape)
        # print('row example: ', res[0])
        archive = df[archive_cols].drop_duplicates()
        data = df[data_cols].drop_duplicates()
        # archive = df[archive_cols].copy()
        # data = df[data_cols].copy()
        # 测试代码
        archive_stat = archive.groupby(['SUBS_NAME', 'SUBS_ID', 'SUBS_ADDR', 'MP_ID'], as_index=False)['MP_NAME'].count()
        data_stat = data.groupby(['MP_ID'], as_index=False)['DATA_DATE'].count()
        res = pd.merge(archive_stat, data_stat, on='MP_ID', how='outer')
        res['index'] = index
        for row in res.to_dict(orient='records'):
            # yield Row(**row)
            res_return.append(Row(**row))
    return res_return


if __name__ == '__main__':
    data_dir = r'D:\Desktop\关口项目\冀北一体化关口数据验证\湖南中台验证'
    # data_dir = r'D:\Project-Workspace\Python-Projects\DataAnalysis\datasets\zshield'
    spark = SparkSession.builder.master('local[*]').appName('HelloSpark').getOrCreate()
    # data_dir = None
    # spark = SparkSession.builder.appName('HelloSpark').getOrCreate()
    print("Spark.version: ", spark.version)
    # print("SparkSession.configs: ", spark.sparkContext.getConf().getAll())
    equip_subs_df, model_cell_df, cell_mp_df, high_df = read_data(spark, data_dir)
    # 如果是从本地csv读入，分区数就是 1
    print("equip_subs_df.partitions: ", equip_subs_df.rdd.getNumPartitions())
    # equip_subs_df.explain()
    # model_cell_df.explain()
    # cell_mp_df.explain()
    # high_df.explain()
    # print('equip_subs_df: ', equip_subs_df.count())
    # print('model_cell_df: ', model_cell_df.count())
    # print('cell_mp_df: ', cell_mp_df.count())
    # print('high_df: ', high_df.count())
    # equip_subs_df.show()

    equip_subs_df = equip_subs_df.select('SUBS_ID', 'SUBS_NAME', 'SUBS_TYPE', 'SUBS_ADDR', 'ASSET_TYPE', 'VOLT_LEVEL',
                                         'RUN_STATUS').withColumnRenamed('VOLT_LEVEL', 'VOLT_LEVEL_subs')
    model_cell_df = model_cell_df.select('OBJ_ID', 'CELL_ID', 'CELL_NO', 'CELL_NAME', 'CELL_TYPE', 'VOLT_LEVEL',
                                         'SUBS_ID', 'ORG_ID')
    cell_mp_df = cell_mp_df.select('MODEL_ID', 'MP_ID', 'MP_NO', 'MP_NAME', 'MP_RATE', 'SG_CODE')\
        .withColumnRenamed('MP_ID', 'MP_ID_origin')\
        .withColumn('MP_ID', expr("substring(MP_ID_origin, 3, length(MP_ID_origin)-1)"))\
        .dropDuplicates()
    # dropDuplicates() 算子里执行了 HashAggregation 操作，所以更改了分区数，默认为 200
    print('cell_mp_df.partitions: ', cell_mp_df.rdd.getNumPartitions())
    model_cell_lines = model_cell_df.where("CELL_TYPE == '03'")
    model_cell_trans = model_cell_df.where("CELL_TYPE == '02'")
    # equip_subs_df.explain()
    # model_cell_df.explain()
    # cell_mp_df.explain()
    # model_cell_lines.explain()
    # model_cell_trans.explain()

    # 开始关联
    # 关联 母线段 和 计量点
    mp_lines_rela = model_cell_lines.join(cell_mp_df, model_cell_lines.OBJ_ID == cell_mp_df.MODEL_ID, 'left')
    # 关联 变压器 和 计量点
    mp_trans_rela = model_cell_trans.join(cell_mp_df, model_cell_trans.OBJ_ID == cell_mp_df.MODEL_ID, 'left')\
        .select('MP_ID', 'OBJ_ID')\
        .withColumnRenamed('OBJ_ID', 'TRAN_OBJ_ID')
    # 这里在 ODPS-spark 上运行时必须要重新读一次数据，不知道为什么
    # cell_mp_df_2 = spark.sql('select MODEL_ID, MP_ID, MP_NO, MP_NAME, MP_RATE, SG_CODE from LOSS_ARCH_REL_CELL_MP')\
    #     .withColumnRenamed('MP_ID', 'MP_ID_origin')\
    #     .withColumn('MP_ID', expr("substring(MP_ID_origin, 3, length(MP_ID_origin)-1)"))\
    #     .dropDuplicates()
    # mp_trans_rela = model_cell_trans.join(cell_mp_df_2, model_cell_trans.OBJ_ID == cell_mp_df_2.MODEL_ID, 'left')\
    #     .select('MP_ID', 'OBJ_ID')\
    #     .withColumnRenamed('OBJ_ID', 'TRAN_OBJ_ID')
    # 关联 变压器 直连表的信息
    subs_group = mp_lines_rela.join(mp_trans_rela, 'MP_ID', 'left')\
        .select('MP_ID', 'OBJ_ID', 'CELL_ID', 'CELL_NO', 'CELL_NAME', 'CELL_TYPE', 'VOLT_LEVEL', 'SUBS_ID', 'ORG_ID',
                'MODEL_ID', 'MP_ID_origin', 'MP_NO', 'MP_NAME', 'MP_RATE', 'SG_CODE', 'TRAN_OBJ_ID')
    # 关联变电站档案
    subs_group_df = subs_group.join(equip_subs_df, 'SUBS_ID', 'left')\
        .select('SUBS_NAME', 'SUBS_ID', 'SUBS_TYPE', 'SUBS_ADDR', 'ASSET_TYPE', 'VOLT_LEVEL_subs', 'RUN_STATUS', 'OBJ_ID',
                'MODEL_ID', 'CELL_ID', 'CELL_NO', 'CELL_NAME', 'CELL_TYPE', 'VOLT_LEVEL', 'ORG_ID', 'MP_ID', 'MP_ID_origin',
                'MP_NO', 'MP_NAME', 'MP_RATE', 'TRAN_OBJ_ID', 'SG_CODE')\
        .withColumnRenamed('OBJ_ID', 'GROUP_ID')
    # subs_group_df_part = mp_lines_rela.join(equip_subs_df, 'SUBS_ID', 'left')
    # subs_group.explain()
    # subs_group_df.explain()
    # subs_group_df_part.show()
    # subs_group.show()

    # 处理电量数据
    high_df = high_df.select('ID', 'D', 'C', 'V')
    high_p_w_fa = high_df.where("C == 'p_w_fa'").withColumnRenamed('V', 'PAP')
    high_p_w_ba = high_df.where("C == 'p_w_ba'").withColumnRenamed('V', 'RAP')
    high_df = high_p_w_fa.join(high_p_w_ba, ['ID', 'D'], 'outer')\
        .withColumnRenamed('D', 'DATA_DATE')\
        .select('ID', 'DATA_DATE', 'PAP', 'RAP')

    # 字段重命名 + 类型转换
    archive = subs_group_df.selectExpr('SUBS_NAME', 'SUBS_ID', 'SUBS_TYPE', 'SUBS_ADDR', 'ASSET_TYPE', 'VOLT_LEVEL_subs', 'RUN_STATUS', 'GROUP_ID',
                'MODEL_ID', 'CELL_ID', 'CELL_NO', 'CELL_NAME', 'CELL_TYPE', 'VOLT_LEVEL', 'ORG_ID', 'MP_ID_origin',
                'MP_NO', 'MP_NAME', 'cast(MP_RATE as FLOAT) as MP_RATE', 'TRAN_OBJ_ID', 'SG_CODE')\
                .withColumnRenamed('VOLT_LEVEL_subs', 'GROUP_VOLT_CODE')\
                .withColumnRenamed('MP_RATE', 'T_FACTOR')\
                .withColumnRenamed('TRAN_OBJ_ID', 'TF_ID')\
                .withColumnRenamed('MP_ID_origin', 'MP_ID')
    data = high_df.selectExpr("ID as MP_ID", "DATA_DATE", "cast(PAP as FLOAT) as PAP_PHI", "cast(RAP as FLOAT) as RAP_PHI")
    # 电量数据关联档案
    archive_data = archive.join(data, archive.MP_ID == data.MP_ID, 'right')
    # high_df_arch = subs_group_df.join(high_df, subs_group_df.MP_ID_origin == high_df.ID, 'right')
    # mp_lines_rela.printSchema()
    # mp_trans_rela.printSchema()
    # subs_group.printSchema()
    # subs_group_df.printSchema()
    # high_df.printSchema()
    # high_df_arch.printSchema()
    # print('subs_group_df.count: ', subs_group_df.count())
    # print('high_df_arch.count: ', high_df_arch.count())
    # mp_lines_rela.show()
    # mp_trans_rela.show()
    # subs_group_df.show()
    # high_df.show()
    # high_df_arch.show()
    # archive.printSchema()
    # data.printSchema()
    # archive_data.printSchema()
    # archive.show()
    # data.show()

    # 调用模型
    # cache 一下，避免重复运算
    archive.cache()
    data.cache()
    archive_data.cache()
    print('data.count: ', data.count())
    print('archive_data.count: ', archive_data.count())
    print('archive_data.partitions: ', archive_data.rdd.getNumPartitions())
    archive_rep = archive_data.repartition(10, 'SUBS_ID')
    print('archive_rep.partitions: ', archive_rep.rdd.getNumPartitions())
    # archive_proc = archive_rep.rdd.mapPartitions(foreach_subs_proc)
    archive_proc = archive_rep.rdd.mapPartitionsWithIndex(foreach_subs_proc)
    archive_proc.cache()
    print('archive_proc.count: ', archive_proc.count())
    # archive_proc.foreach(print)
    # for row in archive_proc.take(5):
    # for row in archive_proc.collect():
    #     print(row)
    archive_proc.toDF().show()

    # 测试代码
    # print('archive.partitions: ', archive.rdd.getNumPartitions())
    # archive_rep = archive.repartition(3, 'SUBS_ID')
    # print('archive_rep.partitions: ', archive_rep.rdd.getNumPartitions())
    # archive_test = archive_rep.select('SUBS_ID', 'SUBS_NAME', 'SUBS_ADDR', 'GROUP_VOLT_CODE', 'MP_ID').rdd.mapPartitions(foreach_subs_test)
    # print("archive_test.__class__: ", archive_test.__class__)
    # print(archive_test.collect())
    # archive_test_2 = archive_rep.select('SUBS_ID', 'SUBS_NAME', 'SUBS_ADDR', 'GROUP_VOLT_CODE', 'MP_ID').rdd.mapPartitions(foreach_subs_test_2)
    # print('archive_test_2.count: ', archive_test_2.count())
    # for row in archive_test_2.take(5):
    #     print(row)

    # spark.stop()

