# %% importing packages
import os
import torch
import warnings
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertTokenizer, BertModel, BertForPreTraining, BertForSequenceClassification
from datasets import load_dataset_builder, get_dataset_config_names, get_dataset_config_info, \
    get_dataset_infos, get_dataset_split_names,  load_dataset
warnings.filterwarnings('ignore')

# 基于 wikitext 语料，训练一个自己的BERT模型
# %% ---------------- 加载数据集 -----------------------
dataset_name = 'wikitext'
# 查看该数据集下的配置，也就是有哪些类型的数据集可供使用
get_dataset_config_names(dataset_name)
# 查看指定数据集的split
get_dataset_split_names(dataset_name, 'wikitext-2-raw-v1')
builder = load_dataset_builder(dataset_name, 'wikitext-2-raw-v1')
# 加载数据集
# data = load_dataset(dataset_name, 'wikitext-2-raw-v1')
print("hello")

# ------------------- 训练 tokenizer ------------------



# ------------------- 训练 BERT 模型 ------------------






# %% 加载 Bert的 配置文件，Tokenizer 和 预训练模型
# base_path = r"/Users/danielzhang/Documents/Python-Projects/DataAnalysis/bert-pretrained-models"
base_path = r"D:\Project-Workspace\Python-Projects\DataAnalysis\bert-pretrained-models"
model_path = os.path.join(base_path, "bert-base-uncased")
config = BertConfig.from_pretrained(pretrained_model_name_or_path=model_path)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
bert_basic = BertModel.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
bert_train = BertForPreTraining.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
# bert_cls = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
# 有了上述的 config 对象之后，可以直接使用上述的 config 对象创建各个模型（但是不能用于tokenizer的实例化），比如
# bert_basic = BertModel(config)
# bert_train = BertForPreTraining(config)
# bert_cls = BertForSequenceClassification(config)

# %% 使用 BertTokenizer
sentence = ["After stealing money from the bank vault",
            "the bank robber was seen fishing on the Mississippi river bank."]
inputs = tokenizer(sentence, padding=True, return_tensors='pt')
# 可以看下处理后的句子还原之后是什么样
# inputs_decode = tokenizer.decode(token_ids=inputs['input_ids'][0])
print('input_ids.shape: ', inputs['input_ids'].shape)
print('token_type_ids.shape: ', inputs['token_type_ids'].shape)
print('attention_mask.shape: ', inputs['attention_mask'].shape)


# %% 使用 BertModel
# 将加载的预训练模型置于 evaluation 状态，这样会关闭其中的 dropout
bert_basic.eval()
with torch.no_grad():
    # 可以直接将 tokenizer 得到的输出作为输入，只要使用拆包技巧就行
    outputs = bert_basic(**inputs, output_hidden_states=True, output_attentions=True)

# 两个句子: batch_size=2, 每个句子序列的长度 sequence_length=14,
# 最后一层的 hidden_size=768 —— 这个由预训练模型的配置决定
print(outputs.last_hidden_state.shape)
# 对应于 [CLS] token 的 embedding，2 个句子，所以返回了两个
print(outputs.pooler_output.shape)
# 查看每一个隐藏层的状态
print(len(outputs.hidden_states))
# Embedding 层的 隐状态
print(outputs.hidden_states[0].shape)
# 第一个 self-attention 层的 隐状态
print(outputs.hidden_states[1].shape)
# 第一个 self-attention 层的 atttention shape
# 2 个句子 batch_size=2, 12 个 heads, sequence_length=14, sequence_length=14
print(outputs.attentions[0].shape)

t = outputs.to_tuple()
t0, t1, t2, t3 = outputs



# %% 使用 BertForPreTraining
outputs = bert_train(**inputs, output_hidden_states=True, output_attentions=True)



# %% 使用 BertForSequenceClassification