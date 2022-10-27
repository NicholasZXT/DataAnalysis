# %% importing packages
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset_builder, get_dataset_config_names, get_dataset_config_info, \
    get_dataset_infos, get_dataset_split_names,  load_dataset
from tokenizers import Tokenizer, normalizers, pre_tokenizers, models, processors, trainers
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertTokenizerFast, DistilBertForMaskedLM, \
    DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast
from transformers import BertConfig, BertTokenizer, BertModel, BertForPreTraining, \
    BertForSequenceClassification, DataCollatorForLanguageModeling

# 忽略警告信息
import warnings
warnings.filterwarnings('ignore')

# ***************** 基于 wikitext 语料，训练一个 Tokenizer *****************
def __Train_tokenizer():
    pass

# %% ---------------- 加载数据集 -----------------------
# 只使用数据集名称，会下载数据处理脚本，速度慢一些，后续各种查询元数据信息也会比较慢
# path = 'wikitext'
# 使用已经下载好的 数据集脚本 会快一些
# path = r'D:\Project-Workspace\Python-Projects\DataAnalysis\datasets\huggingface\wikitext.py'
path = r'.\datasets\huggingface\wikitext.py'
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
# 1. 实例化一个 Tokenizer 类，使用的sub-word分词模型是 WordPiece
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# 2. 规范化（Normalization）步骤：分词，去除大小写，词形还原等
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
# 或者可以手动拼凑更加细节的控制
# tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])

# 3. 预处理（pre-tokenization）步骤：
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# 或者手动进行精细化处理
# tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])

# 4. 配置sub-word分词模型的训练器
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
tokenizer.save(r".\datasets\huggingface\wikitext-2-raw-v1_tokenizer.json")
# 读取
tokenizer = Tokenizer.from_file(r".\datasets\huggingface\wikitext-2-raw-v1_tokenizer.json")
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


# ****************** Fine-tune BERT 模型 ****************
def __Fine_Tune():
    pass

# %% ----------- 加载 Distil-Bert 模型 -----------------
base_path = r"D:\Project-Workspace\Python-Projects\DataAnalysis\bert-pretrained-models"
model_path = os.path.join(base_path, 'distilbert-base-uncased')
# tokenizer = DistilBertTokenizer.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
config = DistilBertConfig.from_pretrained(model_path)
# model = DistilBertModel(config)
# 这里选择的是 Masked Language Model
model_mlm = DistilBertForMaskedLM(config)
# 查看信息
# print(model_mlm.num_parameters())
# print(tokenizer.__class__)
# print(tokenizer.all_special_tokens)
# print(tokenizer.all_special_ids)
# print(tokenizer.is_fast)

# %% ------------ 使用 IMDB 数据集来 Fine-tune ------------
# data_name = 'imdb'
data_name = r'.\datasets\huggingface\imdb.py'
print(get_dataset_config_names(data_name))
print(get_dataset_split_names(data_name))
imdb = load_dataset(data_name, split='unsupervised')

# %% --------------- 处理数据 -----------------------
# 对于 Masked Language Model，在准备训练语料的时候，通常的做法是将 一个batch里 所有样本的文本段落拼在一起，然后按照固定长度分成 chunk，用这些
# chunk 作为训练语料，在这个过程中，还需要对训练语料进行 随机 mask 处理

# 1. 首先使用 tokenizer 进行分词，在分词的过程中，还要新增一个 word_ids，记录下每个batch中，句子token的下标
# examples = imdb['text'][0:5]
# result = tokenizer(examples)
# print(len(result["input_ids"]))
# t = [result.word_ids(i) for i in range(len(result["input_ids"]))]
def tokenize_function(examples):
    # examples 是以 batch 传入的样本
    # result 是分词后的结果(BatchEncoding对象)，包括 input_ids 和 attention_mask 两个key
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        # 新增一个key，记录下每个batch的句子里，各个token的下标
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# 使用 Dataset 对象的 map 方法，以 batch 方式处理
imdb_tokenized = imdb.map(tokenize_function, batched=True, remove_columns=["text", "label"])
print(imdb_tokenized)

# 2. 分词结束后，需要将 一个batch 里的所有样本拼接起来，然后按照固定长度(chunk_size)分割成 chunk
chunk_size = 128
# 使用下面的函数完成操作
def group_texts(examples):
    # examples 是一个 batch 的样本，keys()返回的是 input_ids, attention_mask, word_ids
    # 分别将上述 3 个key在 一个batch 下样本都拼凑到一起
    # sum函数签名为 sum(Iterable, start=0)，examples[k] 是一个 List[List], 使用 sum 时，会对其中的 子list 执行 + ，也就是list的拼接，
    # 最后的结果再 + start，由于 start默认是整数0，无法和 list 执行 +， 所以这里使用了一个空列表
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # 计算拼接后的总长度，list(examples.keys())[0] 就是 input_ids 这个key
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 对于最后一个chunk，它的长度可能小于 chunk_size，这里直接丢弃
    total_length = (total_length // chunk_size) * chunk_size
    # 对拼接后的 token 序列按照 chunk_size 进行分割
    # k 依次是 input_ids, attention_mask, word_ids —— 也就是依然是这 3 个特征
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column —— 这个 label 后面会被作为被 mask 词的预测标签
    result["labels"] = result["input_ids"].copy()
    return result

imdb_chunks = imdb_tokenized.map(group_texts, batched=True)
# 这里可以看出，样本个数已经比原来的 50000 要多了
print(imdb_chunks)

# 3. 对数据集进行分割 chunk 之后，还需要每个chunk中的 token 进行随机 mask，这一步是通过 DataCollatorForLanguageModeling 完成的
# 使用 tokenizer 初始化 DataCollator 对象
imdb_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
# 可以查看下效果
samples = [imdb_chunks[i] for i in range(2)]
# 去除掉其中的 word_ids，因为 collator 不接受这个特征
for sample in samples:
    _ = sample.pop("word_ids")
res = imdb_collator(samples)
# 查看 Mask 的效果
tokenizer.decode(res['input_ids'][0])




# %% ------------------- 训练 BERT 模型 ------------------
# 1. 加载上面训练好的 tokenizer
raw_tokenizer = Tokenizer.from_file(r".\datasets\huggingface\wikitext-2-raw-v1_tokenizer.json")
# 封装成 PreTrainedTokenizerFast对象
custom_tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer)
# print(custom_tokenizer.is_fast)

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