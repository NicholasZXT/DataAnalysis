from elasticsearch import Elasticsearch, helpers
import numpy as np
import pandas as pd

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# es.cat.health(v=True)
# es.cat.indices(v=True)
# es.cat.master()
# es.cluster.health()
# es.close()

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
    # 遍历df的每一行，将每一行变成一个actions
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