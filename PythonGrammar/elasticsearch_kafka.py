import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConflictError, RequestError
from kafka import KafkaClient, KafkaProducer, KafkaConsumer
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


# ========================== kafka ===========================
def __Kafka_Practice():
    pass

bootstrap_servers=['kafka-1:19091', 'kafka-2:19092', 'kafka-3:19093']
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