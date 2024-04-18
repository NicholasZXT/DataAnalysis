import os
import traceback
from socket import timeout
import argparse
from time import sleep, time
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from elasticsearch.exceptions import ConflictError, RequestError
from elasticsearch import helpers, Elasticsearch, RequestError

from kafka import KafkaClient, KafkaProducer, KafkaConsumer, TopicPartition
from kafka.errors import KafkaTimeoutError


# ======================= Elasticsearch ===============================
def __Elasticsearch_Practice():
    pass

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# es.cat.health(v=True)
# es.cat.indices(v=True)
# es.cat.master()
# es.cluster.health()
# es.close()

query = {
    "query": {
        "term": {
            "field-1": "abc"
        }
    }
}

res = es.delete_by_query(index='index-2', body=query)

es.indices.exists(index="new-index-5")
try:
    res = es.indices.create(index="new-index-4")
except RequestError as e:
    print(e.info)
    ex = e
    raise IOError("failed to create index")

body = {
    'field-1': 'data-1',
    'field-2': 'data-2'
}
res1 = es.index(index='index-2', body=body, id='3')
res2 = es.index(index='index-2', body=body, id='3')
res3 = es.index(index='index-2', body=body, id='3', op_type='create')
res4 = es.index(index='index-2', body=body,  op_type='create')

# ----------- 测试更新的冲突 ---------------------
body = {
    'query': {
        'term': {
            '_id': '2'
        }
    }
}
res1 = es.search(body=body, index='index-2', seq_no_primary_term=True)
record = res1['hits']['hits'][0]

update_body_1 = {
    'doc': {
        'field-1': 'new----',
        'field-2': 'ikbc'
    }
}
res2 = es.update(index='index-2', id='2', body=update_body_1)

update_body_2 = {
    'doc': {
        'field-1': 'new-1',
        'field-2': 'ikbc-1'
    }
}
res3 = es.update(index='index-2', id='2', body=update_body_2,if_seq_no=record['_seq_no'], if_primary_term=record['_primary_term'])


index_mapping = {
  "mappings":{
    "properties":{
      "group_col":{
        "type":"keyword"
      },
      "value":{
        "type": "float"
      }
    }
  }
}
es.indices.create("test-index", body=index_mapping)

data_dict = {'group_col':['a', 'a', 'b', 'b', 'b', 'c'], "value": [1, 2, 3, 4, 5, 6]}
df = pd.DataFrame(data_dict)

# t = df.iterrows()
# i, row = next(t)

def df2actions(df, index_name, use_source=True):
    """
    遍历df的每一行，将每一行变成一个actions
    @param df:
    @param index_name:
    @param use_source:
    @return:
    """
    for i, row in df.iterrows():
        source = row.to_dict()
        if use_source:
            action = {
                "_index": index_name,
                "_source": dict(source)
            }
        else:
            action = {
                "_index": index_name,
            }
            action.update(source)
        yield action


#批量导入
# actions = df2actions(df, index_name='test-index-source', use_source=True)
actions = df2actions(df, index_name='test-index-no-source', use_source=False)
actions = list(actions)
res = helpers.bulk(es, actions)
res_parallel = helpers.parallel_bulk(es, actions, chunk_size=2, thread_count=2)

res_iter = iter(res)
item = next(res_iter)


class EsClient:
    def __init__(self, es_host):
        self.es = Elasticsearch(es_host)

    def query(self, index, condition_dict=None, size=500, query_dsl=None, df_format=True):
        """
        封装查询方式， 适用于小批量数据（不超过10000）
        @param index:
        @param condition_dict:
        @param size: 默认500，最大不能超过10000
        @param query_dsl: 自定义的ES查询语句
        @param df_format: 是否转成DataFrame
        @return:
        """
        if size == 500:
            print(f"only return part of results with default size: {size}, please check size parameter.")
        elif size > 10000:
            print(f"parameter size is too large: {size}.")
            raise IOError
        else:
            print(f"return results with limited size: {size}.")
        condition_dict = condition_dict if condition_dict else {}
        if query_dsl is None:
            terms = [{'term': {k: v}} for k, v in condition_dict.items()]
            query_dsl = {
                'query': {
                    'bool': {
                        'must': terms
                    }
                }
            }
        res = self.es.search(index=index, body=query_dsl, request_timeout=20, size=size)
        source_list = []
        for hit in res['hits']['hits']:
            source_list.append(hit['_source'])
        res_final = None
        if df_format:
            res_final = pd.DataFrame(source_list)
        else:
            res_final = source_list
        return res_final

    def scroll_search(self, index, query_dict=None, query_dsl=None, size=500, scroll='5s'):
        """
        滚动查询大批量数据
        @param index:
        @param query_dict: 以 k-v 对 传入的条件
        @param query_dsl: 自定义的查询DLS
        @param size: 每批数据的大小
        @param scroll: 滚动窗口的持续时间，字符串形式
        @return:
        """
        _bool_query = dict()
        condition_dict = query_dict if query_dict else {}
        if query_dsl is None:
            terms = [{'term': {k: v}} for k, v in condition_dict.items()]
            query_dsl = {
                'query': {
                    'bool': {
                        'must': terms
                    }
                }
            }
        try:
            res = self.es.search(index=index, body=query_dsl, size=size, scroll=scroll)
        except Exception as e:
            print("failed to scroll search for index ‘{}’".format(index))
            raise e
        hits = res['hits']['hits']
        total_value = res['hits']['total']['value']
        i = 1
        scroll_num = len(hits)
        while hits:
            yield hits
            # print("scroll batch: ", i)
            i += 1
            res = self.es.scroll(scroll_id=res['_scroll_id'], scroll=scroll)
            hits = res['hits']['hits']
            # print("hits num: ", len(hits))
            scroll_num += len(hits)
        print("Scroll search total value is : {}, and scroll hits are : {}".format(total_value, scroll_num))

    @staticmethod
    def df2actions(df, index_name, id=False):
        """
        将 DataFrame 中的每条记录转成 action 格式，以便 ES helper 接口批量导入
        @param df:
        @param index_name: 要写入的 ES 索引名称
        @return:list of dict, 每一个dict是一个action
        @param id: 是否手动设置_id
        """
        count = 1
        # 处理缺失值，np.nan 无法写入ES
        df = df.where(pd.notnull(df), None)
        for i, row in df.iterrows():
            if count > 0 and count % 5000 == 0:
                print(f'import data count is {count}...')
            count += 1
            source = row.to_dict()
            action = {
                "_index": index_name,
                # ES7.0 以后不需要
                # "_type": '_doc',  # if self.es7 else 'medvol',
                '_source': source
            }
            if id:
                action['_id'] = source['LINE_ID'] + '-' + source['DATA_DATE']
            yield action

    def do_bulk(self, actions):
        """
        执行多线程批量导入ES
        @param actions: 要导入的 ES actions，最好是生成器的形式
        @return:
        """
        success = 0
        failed = 0
        write_success = True
        while write_success and actions:
            write_success = False
            try:
                for res in helpers.parallel_bulk(self.es, actions, chunk_size=50, request_timeout=10000, thread_count=2):
                    # res 是一个 长度=2 的tuple，第一个元素是True或者False，表示是否导入成功
                    if success > 0 and success % 10000 == 0:
                        print("do_bulk running, success for {} records".format(success))
                    if res[0]:
                        success += 1
                    else:
                        failed += 1
                        print('do_bulk failed res:%s' % res)
            except timeout:
                # 写入超时
                print('bulk timeout and need retry')
                actions = actions[success:]
                sleep(60)
                # 打开写入开关，在下一个while循环中继续写入
                write_success = True
            except Exception as e:
                print('---------------Caught do_bulk exception-------------------')
                print(e)
                # traceback.print_exc()
            else:
                break
        # 返回 写入成功的记录数、总记录数
        print(f"bulk with success records: {success}, failed records: {failed}, total records: {success+failed}.")
        return [success, success + failed]

    def write_df(self, df, index, id=False):
        """
        将 pandas.DataFrame 批量导入 ES 中的指定索引，如果索引不存在，则创建一个新索引
        @param df:
        @param index:
        @return:
        """
        actions = self.df2actions(df, index, id=id)
        if self.es.indices.exists(index):
            return self.do_bulk(actions)
        else:
            self.es.indices.create(index=index)
            print("creating new index '{}'.".format(index))
            return self.do_bulk(actions)

    def export_data(self, index, query_dsl, path, scroll='10s'):
        if not os.path.exists(path):
            print(f"please check whether path ‘{path}’ existence.")
            return None
        file_path = os.path.join(path, index.strip("*") + '.csv')
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"file_path ‘{file_path}’ already exists and it was deleted !!!")
        # source_list = []
        records_iter = self.scroll_search(index=index, query_dsl=query_dsl, scroll=scroll)
        sum = 0
        for records in records_iter:
            source_list = [record['_source'] for record in records]
            df = pd.DataFrame(source_list)
            if sum == 0:
                df.to_csv(file_path, index=False)
            else:
                df.to_csv(file_path, mode='a', index=False, header=False)
            sum += len(source_list)
        # with open(file_path, 'a') as f:
        #     for records in records_iter:
        #         for record in records:
        #             # f.write(json.dumps(record['_source']))
        #             f.write(json.dumps(record['_source'], ensure_ascii=False))
        #             f.write("\n")
        #             sum += 1
        print(f"export {sum} records to file: {file_path}.")

    def reindex(self, index_map, source_host, source_user, source_pw, max_docs=20000):
        body = {
            "source": {
                "remote": {
                    "host": source_host,
                    "username": source_user,
                    "password": source_pw
                },
                "size": 1000,
                "index": None
            },
            "dest": {
                "index": None,
                "op_type": "create"
            }
        }
        res = {}
        for k, v in index_map.items():
            if not self.es.indices.exists(k):
                print(f"target index {v} not exists, create it.")
                self.es.indices.create(v)
            print(f"reindex from '{k}' to '{v}'.")
            body['source']['index'] = k
            body['dest']['index'] = v
            temp = self.es.reindex(body, timeout='1m')
            res[k] = temp
        return res



# ========================== kafka ===========================
def __Kafka_Practice():
    pass

bootstrap_servers = ['kafka-1:19091', 'kafka-2:19092', 'kafka-3:19093']
# --------------- 生产者 ----------------
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
# 消费者配置列表
producer.config
# 连接状态
producer.bootstrap_connected()
# 发送消息，异步发送，返回一个 FutureRecordMetadata 对象
res_future = producer.send(topic='my-topic', value="message-1".encode())
type(res_future)   # kafka.producer.future.FutureRecordMetadata
# value 是对要封装的 kafka.producer.future.RecordMetadata 对象，它封装了要发送的消息
res_future.value
# 检查是否完成，返回 bool 值
res_future.is_done
# 检查是否产生异常
res_future.exception
# 检查是否发送失败，返回 bool 值
res_future.failed()
# get 方法获取发送的结果，可能会阻塞
res = res_future.get()
# 发送的 topic, partition, offset
res.topic
res.partition
res.offset
# 关闭生产者
producer.close()


# 写入多条数据
messages_list = ['ms-1', 'ms-2', 'ms-3', 'ms-4', 'ms-5']
producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
for message in messages_list:
    producer.send(topic='my-topic', value=message.encode())
producer.close()

# -------------- 消费者 -----------------------
# topic 是 str 或者 list of str，指定topic，注意，这个参数没有参数名，传参时必须在第一个
# 如果初始化时没有设置topic，后续应该通过 subscrbe() 方法订阅
# consumer = KafkaConsumer('my-topic', bootstrap_servers=bootstrap_servers)
# consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
consumer = KafkaConsumer(group_id='my-group-1', bootstrap_servers=bootstrap_servers)
# 检查连接状态
consumer.bootstrap_connected()
# 查看 可供订阅 的 topics
consumer.topics()
# 订阅一个topic
consumer.subscribe(topics=['my-topic'])
# 查看当前消费者订阅的 topic
consumer.subscription()
# 检查 topic 有哪些 partition 可以消费, 返回的是一个 set
pars = consumer.partitions_for_topic('my-topic')
pars
# par = pars.pop()
# 手动分配给消费者一个partition，必须以 list of TopicPartition 的形式——使用了 subscribe() 之后不能使用此方法
# consumer.assign(partitions=[0])
# 获取分配给当前消费者的partition
consumer.assignment()
# 获取下一个消息的 offset
# consumer.position()
# 重置 offset 到开头
consumer.seek_to_beginning()

# 获取消息
# 这里要使用 auto_offset_reset='earliest' 参数，否则只会读取最新产生的数据，读不到已经产生的数据
# 并且如果要消费旧数据的话，每次都要改变 group_id 或者 直接不设置 group_id
# consumer = KafkaConsumer('my-topic', group_id='my-group-3', bootstrap_servers=bootstrap_servers, auto_offset_reset='earliest')
consumer = KafkaConsumer('my-topic', bootstrap_servers=bootstrap_servers, auto_offset_reset='earliest')
# 有两种消费方式
# 第 1 种，没有获取到消息时会阻塞
# 可以在上面的参数里设置 consumer_timeout_ms=2000 参数，表示等待 2s 后没有消费到数据就停止，跳出阻塞
for message in consumer:
    # message 直接就是 kafka.consumer.fetcher.ConsumerRecord 类实例
    print("message.class: ", type(message))
    print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition, message.offset, message.key, message.value))

# 第 2 种
consumer = KafkaConsumer('my-topic', bootstrap_servers=bootstrap_servers, auto_offset_reset='earliest')
# 必须要设置超时参数，返回值是一个 dict
records_dict = consumer.poll(timeout_ms=1000)
# k1 是 kafka.structs.TopicPartition 对象
k1 = list(records_dict.keys())[0]
# v1 是 list of kafka.consumer.fetcher.ConsumerRecord, 每个元素是获取到的消息
v1 = records_dict[k1]
record = v1[0]

# 关闭消费者——这个方法重复调用也不会引发异常
consumer.close()


def kafka_export(topic, out_dir):
    consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, auto_offset_reset='earliest', consumer_timeout_ms=2000)
    total = 0
    cols = []
    if not os.path.exists(out_dir):
        print(f"out_dir '{out_dir}' not exist, please check !")
        raise FileNotFoundError(f"out_dir '{out_dir}' not exist, please check !")
    out_file = os.path.join(out_dir, topic+'.tsv')
    with open(out_file, mode='w', encoding='utf-8') as file:
        for message in consumer:
            # message 直接就是 kafka.consumer.fetcher.ConsumerRecord 类实例
            # print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition, message.offset, message.key, message.value.decode()))
            value = message.value.decode()
            data = json.loads(value)
            print(data)
            if len(cols) == 0:
                cols = list(data.keys())
                for col in cols:
                    file.write(col + '\t')
                file.write('end_col\n')
            for col in cols:
                file.write(str(data.get(col, '')) + '\t')
            file.write('end\n')
            total += 1
    consumer.close()
    print(f"export total: {total}")

topic = 'my-topic'
kafka_export(topic, os.getcwd())
# 读取方式
# df = pd.read_csv(os.path.join(os.getcwd(), topic+'.tsv'), delimiter='\t', header=0, na_values='None')
# print(df.head(3))

def count_topic_records(bootstrap_servers, topic: str):
    """
    统计指定topic下，所有partition里的数据记录总数
    """
    consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
    # 获取指定topic的所有partition id
    partition_ids = list(consumer.partitions_for_topic(topic))
    # 构造TopicPartition对象
    partitions = [TopicPartition(topic, par_id) for par_id in partition_ids]
    offset_begin = consumer.beginning_offsets(partitions=partitions)
    offset_end = consumer.end_offsets(partitions=partitions)
    # offset_begin或offset_end是一个dict，key是TopicPartition对象，value是offset
    total = 0
    for par in partitions:
        par_begin = offset_begin[par]
        par_end = offset_end[par]
        par_record_num = par_end - par_begin
        total += par_record_num
        print(f"topic '{topic}' partition@[{par.partition}]: offset_start={par_begin}, offset_end={par_end}, record_num={par_record_num}.")
    print(f"total records in topic '{topic}' of {len(partitions)} partitions: {total}.")
    return total

bootstrap_servers = ['hadoop101:9092', 'hadoop102:9092', 'hadoop103:9092']
topic = 'my-topic'
count_topic_records(bootstrap_servers, topic)


def now_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def topic_summary(consumer: KafkaConsumer, topic: str, timeout: int=0):
    # topic = 'tyc_dishonest_person_integrated_circuit'
    # 获取指定topic的所有partition id
    partition_ids = list(consumer.partitions_for_topic(topic))
    # 构造TopicPartition对象
    partitions = [TopicPartition(topic, par_id) for par_id in partition_ids]
    offset_start = consumer.beginning_offsets(partitions=partitions)
    offset_end = consumer.end_offsets(partitions=partitions)
    # offset_begin或offset_end是一个dict，key是TopicPartition对象，value是offset
    total = 0
    start_time, end_time = None, None
    # print(f"[{now_time()}] {topic} -> before for loop...")
    for par in partitions:
        par_start = offset_start[par]
        par_end = offset_end[par]
        par_record_num = par_end - par_start
        total += par_record_num
        print(f"[{now_time()}] topic '{topic}' partition@[{par.partition}]: offset_start={par_start}, offset_end={par_end}, record_num={par_record_num}.")
        # 下面这个判断很重要，对于空的partition，默认下会一直阻塞 ----- KEY
        # 如果设置了 consumer_timeout_ms，则该topic每次查询此partition都会超时，此时如果设置了重试逻辑，则会一直重试，直到超出最大次数
        if par_record_num == 0:
            print(f"[{now_time()}] topic '{topic}' partition@[{par.partition}]: empty partition, skip query timestamp...")
            continue
        consumer.assign([par])
        rec_start, rec_end = None, None
        # print(f"[{now_time()}] {topic} -> for loop -> partition[{par.partition}] -> seek_to_beginning ...")
        # 获取当前partition最早的记录时间
        consumer.seek_to_beginning()
        r = consumer.poll(max_records=1, timeout_ms=timeout)
        # print(f"[{now_time()}] {topic} -> for loop -> partition[{par.partition}] -> seek_to_beginning -> scroll ...")
        for rec in consumer:
            rec_start = rec
            break
        # print(f"[{now_time()}] {topic} -> for loop -> partition[{par.partition}] -> seek_to_end ...")
        # 获取当前partition最新记录的时间
        consumer.seek(partition=par, offset=max(par_end-1, 0))
        # 另一种方式
        # consumer.seek_to_end()
        # last_pos = consumer.position(par)
        # consumer.seek(partition=par, offset=last_pos-1)
        r = consumer.poll(max_records=1, timeout_ms=timeout)
        # print(f"[{now_time()}] {topic} -> for loop -> partition[{par.partition}] -> seek_to_end -> scroll ...")
        for rec in consumer:
            rec_end = rec
            break
        # 有了上面的 if par_record_num == 0 的判断，这里检查就没有必要了
        # if rec_start is None or rec_end is None:
        #     print(f"[{now_time()}] topic '{topic}' partition@[{par.partition}]: failed to query record time...")
        #     raise TimeoutError
        par_start_time = datetime.fromtimestamp(rec_start.timestamp/1000)
        par_end_time = datetime.fromtimestamp(rec_end.timestamp/1000)
        print(f"[{now_time()}] topic '{topic}' partition@[{par.partition}]: "
              f"offset_start={par_start} at {par_start_time.strftime('%Y-%m-%d %H:%M:%S')}, "
              f"offset_end={par_end} at {par_end_time.strftime('%Y-%m-%d %H:%M:%S')}, record_num={par_record_num}.")
        if start_time and start_time <= par_start_time:
            pass
        else:
            start_time = par_start_time
        if end_time and par_end_time <= end_time:
            pass
        else:
            end_time = par_end_time
        # sleep(0.2)
    start_time = start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else None
    end_time = end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else None
    print(f"[{now_time()}] topic '{topic}' of {len(partitions)} partitions has total records : {total}, start at {start_time}, end at {end_time}.")
    return {'topic': topic, 'records': total, 'start': start_time, 'end': end_time}

def kafka_topics_stat_v1(topics):
    # consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
    consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers, consumer_timeout_ms=5*1000)
    topic_num = len(topics)
    res = []
    for num, topic in enumerate(topics, start=1):
        print(f"[{now_time()}] ******** [{num}] querying {topic} ********")
        topic_stat = topic_summary(consumer, topic)
        # topic_stat = topic_summary(consumer, topic, 5*1000)
        res.append(topic_stat)
        print(f"[{now_time()}] ========= [{num}] querying {topic} done, remaining {topic_num-num} =========")
        sleep(1)
    consumer.close()
    topic_stat_df = pd.DataFrame(res)
    topic_stat_df.to_excel("topics_summary.xlsx", index=False)


def kafka_topics_stat_v2(topics):
    """对应抛出 TimeoutError 的失败重试版本，留作对比参考"""
    consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers, consumer_timeout_ms=5 * 1000)
    topic_num = len(topics)
    res = []
    num, retry, max_retry = 1, 0, 30
    while len(topics) > 0:
        # 最大重试次数
        if retry >= max_retry:
            print()
            print(f"[{now_time()}] ------ failed too many times and exceeds max iteration[{max_retry}], stop running ------")
            print(f"remaining topics: {topics}")
            break
        topic = topics.pop(0)
        print(f"[{now_time()}] ******** [{num}] querying {topic} ********")
        # consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers, consumer_timeout_ms=5*1000)
        try:
            topic_stat = topic_summary(consumer, topic)
            # topic_stat = topic_summary(consumer, topic, 5*1000)
            res.append(topic_stat)
            print(f"[{now_time()}] ========= [{num}] querying {topic} done, remaining {topic_num-num} =========")
            num += 1
        except TimeoutError as e:
            # 失败重试放回
            retry += 1
            topics.append(topic)
            print(f"[{now_time()}] ------ [{num}] querying {topic} failed. total retry num: {retry} ------")
        finally:
            # consumer.close()
            sleep(1)
    consumer.close()
    topic_stat_df = pd.DataFrame(res)
    topic_stat_df.to_excel("topics_summary.xlsx", index=False)


consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers)
topics = consumer.topics()
consumer.close()
topic_num = len(topics)
print(f"total topic num to query: {topic_num}...")
kafka_topics_stat_v1(topics)
# kafka_topics_stat_v2(topics)
