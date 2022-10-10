# %% importing packages
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertTokenizer, BertModel, BertForPreTraining, BertForSequenceClassification


# %% 加载 Bert的 配置文件，Tokenizer 和 预训练模型
base_path = r"/Users/danielzhang/Documents/Python-Projects/DataAnalysis/bert-pretrained-models"
model_path = os.path.join(base_path, "bert-base-uncased")
config = BertConfig.from_pretrained(pretrained_model_name_or_path=model_path)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
bert_basic = BertModel.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
# bert_train = BertForPreTraining.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
# bert_cls = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
# 有了上述的 config 对象之后，可以直接使用上述的 config 对象创建各个模型（但是不能用于tokenizer的实例化），比如
# bert_basic = BertModel(config)
# bert_train = BertForPreTraining(config)
# bert_cls = BertForSequenceClassification(config)

# %% 使用 BertTokenizer
s1 = "this is a short sentence"
s2 = "This is a rather longer sequence"
s3 = "Hello, my dog is cute"
sentence = [s1, s2, s3]
inputs = tokenizer(sentence, padding=True, return_tensors='pt')
# 处理后的句子
inputs_decode = tokenizer.decode(token_ids=inputs['input_ids'][0])


# %% 使用 BertModel
outputs = bert_basic(**inputs)



# %% 使用 BertForPreTraining



# %% 使用 BertForSequenceClassification