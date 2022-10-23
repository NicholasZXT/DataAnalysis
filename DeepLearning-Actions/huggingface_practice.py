# %% importing packages
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset_builder, get_dataset_config_names, get_dataset_config_info, \
    get_dataset_infos, get_dataset_split_names,  load_dataset
from tokenizers import normalizers, pre_tokenizers, models, processors, trainers, decoders, Tokenizer
from transformers import BertConfig, BertTokenizer, BertModel, BertForPreTraining, \
    BertForSequenceClassification, DataCollatorForLanguageModeling

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

# ***************** 基于 wikitext 语料，训练一个自己的BERT模型 *****************

# %% ---------------- 加载数据集 -----------------------
# 只使用数据集名称，会下载数据处理脚本，速度慢一些，后续各种查询元数据信息也会比较慢
# path = 'wikitext'
# 使用已经下载好的 数据集脚本 会快一些
path = r'D:\Project-Workspace\Python-Projects\DataAnalysis\datasets\huggingface\wikitext.py'
# 查看该数据集下的配置，也就是有哪些子数据集可供使用
get_dataset_config_names(path)
# 查看指定子数据集的split
get_dataset_split_names(path, 'wikitext-2-raw-v1')
# 加载指定子数据集（必须指定到子数据集名称）的Builder对象
builder = load_dataset_builder(path, 'wikitext-2-raw-v1')
# 加载数据集，上述的脚本会下载数据集，存放在缓存文件夹中
data = load_dataset(path, 'wikitext-2-raw-v1')
data_train = data['train']

# 将数据集组织成生成器，每次返回一个batch的数据
def get_training_corpus(text_data):
    for i in range(0, len(text_data), 1000):
        yield text_data[i: i + 1000]["text"]


# %% ------------------- 训练 tokenizer ------------------
# 这部分基本是参照huggingface的官方教程实现的
# [Building a WordPiece tokenizer from scratch](https://huggingface.co/course/chapter6/8?fw=pt#building-a-wordpiece-tokenizer-from-scratch)
# 一个 tokenizer 其实也是 pipeline，里面包含了多个步骤。
# 1. 实例化一个 Tokenizer 类，使用的词向量模型是 WordPiece
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# 2. 规范化（Normalization）步骤：分词，去除大小写，词形还原等
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
# 或者可以手动拼凑更加细节的控制
# tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])

# 3. 预处理（pre-tokenization）步骤：
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# 或者手动进行精细化处理
# tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])

# 4. 配置词向量模型的训练器
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

# 5. 使用语料进行训练
tokenizer.train_from_iterator(get_training_corpus(data_train), trainer=trainer)

# 6. 后处理（post-processing）流程：也就是在句子开头加上 [CLS]，句子中间和末尾加上 [SEP]
# 首先获取 [CLS] 和 [SEP] 的 token_id
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
# 然后增加后处理流程
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

# 7. 保存训练的 tokenizer
tokenizer.save(r"datasets\huggingface\tokenizer.json")

# 测试训练的 tokenizer
single_sen = "Let's test my pre-tokenizer."
pair_sen = "Let's test this tokenizer...", "on a pair of sentences."
single_encoding = tokenizer.encode(single_sen)
pair_encoding = tokenizer.encode(*pair_sen)
print(single_encoding.tokens)
print(single_encoding.type_ids)
print(single_encoding.attention_mask)
print(pair_encoding.tokens)
print(pair_encoding.type_ids)
print(pair_encoding.attention_mask)


# %% ------------------- 训练 BERT 模型 ------------------
# 1. 加载上面训练好的 tokenizer
custom_tokenizer = Tokenizer.from_file(r"datasets\huggingface\tokenizer.json")

# 2. 初始化 BERT 模型
vocab_size = custom_tokenizer.get_vocab_size()
custom_config = BertConfig(vocab_size=vocab_size)
custom_bert = BertForPreTraining(custom_config)

# 3. 设置训练数据生成器
data_collator = DataCollatorForLanguageModeling()



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