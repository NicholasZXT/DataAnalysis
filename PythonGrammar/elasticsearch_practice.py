from elasticsearch import Elasticsearch
from elasticsearch import helpers
import numpy as np
import pandas as pd

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es.cat.health(v=True)
es.cat.indices(v=True)

es.cluster.health()
es.close()

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

data_dict = {'group_col':['a', 'a', 'b', 'b', 'b', 'c'], "value": [1,2,3,4,5,6]}
df = pd.DataFrame(data_dict)

t = df.iterrows()
i, row = next(t)

def df2actions(df, index_name):
    # 遍历df的每一行，将每一行变成一个actions
    for i, row in df.iterrows():
        source = row.to_dict()
        action = {
            "_index": index_name,
        }
        action.update(source)
        yield action


#批量导入
actions = df2actions(df, index_name='test-index')
res = helpers.bulk(es, actions)