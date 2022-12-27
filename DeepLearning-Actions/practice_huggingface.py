# %% importing packages
import os
from time import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import load_dataset_builder, get_dataset_config_names, get_dataset_split_names, load_dataset
from tokenizers import Tokenizer, normalizers, pre_tokenizers, models, processors, trainers
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask, PreTrainedTokenizerFast
from transformers import BertConfig, BertTokenizerFast, BertModel, BertForMaskedLM, BertForNextSentencePrediction, BertForPreTraining
from transformers import TextDatasetForNextSentencePrediction
# 忽略警告信息
# import warnings
# warnings.filterwarnings('ignore')

LOCAL_DATA_PATH = r"D:\Project-Workspace\Python-Projects\DataAnalysis\local-datasets"
LOCAL_MODELS_PATH = r"D:\Project-Workspace\Python-Projects\DataAnalysis\bert-pretrained-models"

# %% ------------ 使用huggingface-hub上的 IMDB 数据集来 Fine-tune ------------
# data_name = 'imdb'
data_name = os.path.join(LOCAL_DATA_PATH, 'huggingface', 'imdb.py')
print(get_dataset_config_names(data_name))
print(get_dataset_split_names(data_name))
imdb = load_dataset(data_name, split='unsupervised')
print(imdb)


# ============== 使用 Masked Language Model 来 fine-tune BERT 模型 ==============
def __Fine_tune_by_MLM():
    pass

# %% ----------- Step 1: 加载 Bert 模型 -----------------
# BERT-base模型
model_path = os.path.join(LOCAL_MODELS_PATH, 'bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path)
# 需要手动设置输出 attentions 和 hidden_states
# config.output_attentions = True
# config.output_hidden_states = True
model = BertModel(config)
model_mlm = BertForMaskedLM(config)

# 查看信息
# print(model_mlm.num_parameters())
# print(tokenizer.__class__)
# print(tokenizer.all_special_tokens)
# print(tokenizer.all_special_ids)
# print(tokenizer.is_fast)


# %% --------------- Step 2: 处理数据 -----------------------
# 对于 Masked Language Model，在准备训练语料的时候，通常的做法是将 一个batch里 所有样本的文本段落拼在一起，然后按照固定长度分成 chunk，用这些
# chunk 作为训练语料，在输入模型之前，还需要对训练语料进行 随机 mask 处理

# (1) 首先使用 tokenizer 进行分词，在分词的过程中，还要新增一个 word_ids，记录下每个batch中，句子token的下标
# examples = imdb['text'][0:5]
# result = tokenizer(examples)
# print(len(result["input_ids"]))
# t = [result.word_ids(i) for i in range(len(result["input_ids"]))]
def tokenize_function(examples):
    # examples 是以 batch 传入的样本
    # result 是分词后的结果(BatchEncoding对象)，包括 input_ids 和 attention_mask 两个key
    result = tokenizer(examples["text"])
    # 如果后面直接使用 DataCollatorForWholeWordMask，那就不需要这个 word_ids
    # if tokenizer.is_fast:
        # 新增一个key，记录下每个batch的句子里，各个token的下标
        # result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# 使用 Dataset 对象的 map 方法，以 batch 方式处理
imdb_tokenized = imdb.map(tokenize_function, batched=True, remove_columns=["text", "label"], load_from_cache_file=True)
# 可以看出，样本数量没变，但是 features 由 ['text', 'label'] --> ['input_ids', 'attention_mask', 'word_ids']
# print(imdb_tokenized)

# (2) 分词结束后，需要将 一个batch 里的所有样本拼接起来，然后按照固定长度(chunk_size)分割成 chunk
chunk_size = 128
# chunk_size = 16
# 使用下面的函数完成操作
def group_texts(examples):
    # examples 是一个 batch 的样本，keys()返回的是 input_ids, attention_mask, word_ids
    # 分别将上述 3 个key在 一个batch 下样本都拼凑到一起
    # sum函数签名为 sum(Iterable, start=0)，examples[k] 是一个 List[List], 使用 sum 时，会对其中的 子list 执行 + ，也就是list的拼接，
    # 最后的结果再 + start，由于 start默认是整数0，无法和 list 执行 +， 所以这里使用了一个空列表
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # 计算此 batch 的样本拼接后的总长度，list(examples.keys())[0] 就是 input_ids 这个key
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 对于最后一个chunk，它的长度可能小于 chunk_size，这里直接丢弃
    total_length = (total_length // chunk_size) * chunk_size
    # 对拼接后的 token 序列按照 chunk_size 进行分割
    # k 依次是 input_ids, attention_mask, word_ids —— 也就是依然是这 3 个特征
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column —— 这个 label 后面会作为 mask 词的预测标签
    result["labels"] = result["input_ids"].copy()
    return result

imdb_chunks = imdb_tokenized.map(group_texts, batched=True, load_from_cache_file=True)
# 这里可以看出，样本个数已经比原来的 50000 要多了
print(imdb_chunks)
# 查看一下样本内容
# print(imdb_chunks['input_ids'][0])
# print(tokenizer.decode(imdb_chunks['input_ids'][0]))
# print(imdb_chunks['labels'][0])
# print(tokenizer.decode(imdb_chunks['labels'][0]))

# (3) 对数据集进行分割 chunk 之后，还需要每个chunk中的 token 进行随机 mask，这一步是通过 DataCollatorForLanguageModeling 完成的
# 使用 tokenizer 初始化 DataCollator 对象，因为要使用 tokenizer 中的特殊token来进行mask
# 下面这个只能 mask 分词后的 subword 对应的 token
# lm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
# 下面这个可以 mask 分词后，多个 subword 对应的 token 组成的 完整word，使用这个时，不需要上面生成的 word_ids
lm_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=0.15)
# 可以查看下效果
# samples = [imdb_chunks[i] for i in range(2)]
# 去除掉其中的 word_ids，因为word_ids的开头或者结尾是 None，无法转成 tensor，会报错
# for sample in samples:
#     _ = sample.pop("word_ids")
# res = lm_collator(samples)
# res 是 transformers.tokenization_utils_base.BatchEncoding 对象
# print(res.__class__)
# 查看 Mask 的效果
# print(res['input_ids'][0])
# print(tokenizer.decode(res['input_ids'][0]))
# labels 中，除了被 mask 的 token， 其他位置的 token 都被转成了 -100
# print(res['labels'][0])

# (4) 将上述的数据封装成 pytorch 的 DataLoader
imdb_dataloader = DataLoader(dataset=imdb_chunks, batch_size=16, shuffle=True, collate_fn=lm_collator)


# %% ---------------- Step 3:  Fine-Tune 模型 ------------------------
# 查看下MLM模型输出
# test_batch_size = 4
# sample = [imdb_chunks[i] for i in range(test_batch_size)]
# sample_batch = lm_collator(sample)
# with torch.no_grad():
#     output = model(sample_batch['input_ids'])
#     output_mlm = model_mlm(**sample_batch)
# print(type(output), '; ', type(output_mlm))
# # 序列中每个 token 的词向量长度由 BERT 的配置决定，同时词典中的token个数决定了MLM输出的向量维度
# print(f"config.hidden_size: {config.hidden_size}, config.vocab_size: {config.vocab_size}")
# # 输入一个 batch 的数据，样本数为 test_batch_size，每个样本中序列长度为 chunk_size
# print(sample_batch['input_ids'].shape)
# # labels 的形状和输入的 input_ids 是一样的
# print(sample_batch['labels'].shape)
#
# # 经过 Bert 模型后的隐藏层输出为 (batch_size,seq_length, hidden_size), 其中 seq_length = chunk_size
# print(output.last_hidden_state.shape)
# # print(output.attentions[-1].shape)
# # print(output.hidden_states[-1].shape)
# # Bert模型最后一层的输出，经过 BertOnlyMLMHead 转换后，每个 token 的维度从 hidden_size --> vocab_size，也就是对应于 vocab 中各个词的概率
# print(output_mlm.logits.shape)
# # 直接输出一个 batch 下，所有样本序列的 交叉熵损失之和，可以直接使用
# print(output_mlm.loss)

# GPU 设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_mlm.cuda(device)
# print(device)
# 定义优化器
optimizer = optim.SGD(params=model_mlm.parameters(), lr=0.1)
# 进行训练
num_train_epochs = 10
loss = 0
print(f"training on device: {device}")
start_time = time()
for epoch in range(1, num_train_epochs+1):
    model_mlm.train()
    for batch_num, batch in enumerate(imdb_dataloader, start=1):
        input_ids, labels = batch['input_ids'], batch['labels']
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # outputs = model_mlm(**batch)
        outputs = model_mlm(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        if batch_num % 100 == 0:
            print(f"training loss at batch '{batch_num}' of epoch '{epoch}' is: {loss}")
        loss.backward()
        optimizer.step()
    print(f"training loss finally at epoch '{epoch}' is : {loss}")
end_time = time()
total_time = end_time - start_time
print(f"total training time is : {total_time}")

# RTX 2070 Super-8G, batch_size=30, chunk_size=128, 1个epoch耗时 1574.86 秒
# RTX 3060-12G, batch_size=30, chunk_size=128, 1个epoch耗时 1854.94 秒, loss=6.076962471008301
# RTX 3060-12G, batch_size=60, chunk_size=128, 1个epoch耗时 1861.36 秒, loss=6.684980869293213




# ============== 使用 Next Sentence Prediction 来 fine-tune BERT 模型 ==============
def __Fine_tune_by_NSP():
    pass

# %% -------------- Step 1: 加载模型 ----------------
model_path = os.path.join(LOCAL_MODELS_PATH, 'bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path)
# 需要手动设置输出 attentions 和 hidden_states
# config.output_attentions = True
# config.output_hidden_states = True
# model = BertModel(config)
model_nsp = BertForNextSentencePrediction(config)


# %% -------------- Step 2: 处理数据 -----------------
# TODO 这里没有现成的数据可供使用，操作比较麻烦
def nsp_tokenize(examples):
    pass

# %% --------------- Step 3: Fine-Tune 模型 ------------------




# ============== 基于 wikitext 语料，训练一个 Tokenizer ==============
def __Training_tokenizer_on_wikitext():
    pass

# %% ---------------- 加载数据集 -----------------------
# 只使用数据集名称，会下载数据处理脚本，速度慢一些，后续各种查询元数据信息也会比较慢
# path = 'wikitext'
# 使用已经下载好的 数据集脚本 会快一些
path = os.path.join(LOCAL_DATA_PATH, 'huggingface', 'wikitext.py')
# 查看该数据集下的配置，也就是有哪些子数据集可供使用
get_dataset_config_names(path)
# 选定子数据集
# split = 'wikitext-2-raw-v1'
split = 'wikitext-103-raw-v1'
# 查看指定子数据集的split
get_dataset_split_names(path, split)
# 加载指定子数据集（必须指定到子数据集名称）的Builder对象
builder = load_dataset_builder(path, split)
# 加载数据集，上述的脚本会下载数据集，存放在缓存文件夹中
wikitext = load_dataset(path, split)
wikitext_train = wikitext['train']
# wikitext_test = wikitext['test']
print(wikitext)
for i in range(5):
    print(f'example {i}:\n', wikitext_train['text'][i])

# 将数据集组织成生成器，每次返回一个batch的数据
def generate_training_corpus(text_data):
    for i in range(0, len(text_data), 1000):
        # yield text_data[i: i + 1000]["text"]
        # 去除掉那些空白行
        yield [text for text in text_data[i: i + 1000]["text"] if len(text) > 1]


# %% ------------------- 训练 tokenizer ------------------
# 这部分基本是参照huggingface的官方教程实现的
# [Building a WordPiece tokenizer from scratch](https://huggingface.co/course/chapter6/8?fw=pt#building-a-wordpiece-tokenizer-from-scratch)
# 一个 tokenizer 其实也是 pipeline，里面包含了多个步骤。
# 1. 实例化一个 Tokenizer 类，使用的sub-word分词模型是 WordPiece, [UNK] 指定的是未知的token对应的符号
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# 2. 规范化（Normalization）步骤：分词，去除大小写，词形还原等
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
# 或者可以手动拼凑更加细节的控制
# tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])

# 3. 预处理（pre-tokenization）步骤：
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# 或者手动进行精细化处理
# tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])

# 4. 配置sub-word分词模型的训练器，指定其中用到的特殊token
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=32000, special_tokens=special_tokens, show_progress=True)

# 5. 使用语料进行训练
tokenizer.train_from_iterator(generate_training_corpus(wikitext_train), trainer=trainer)

# 6. 后处理（post-processing）流程：也就是在句子开头加上 [CLS]，句子中间和末尾加上 [SEP]
# 首先获取 [CLS] 和 [SEP] 的 token_id
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
# 然后增加后处理流程， 主要是指定单个句子、句子对的处理模板，并且还要指定其中用到的特殊token
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

# 7. 保存训练的 tokenizer
tokenizer_path = os.path.join(LOCAL_DATA_PATH, 'huggingface', f"{split}_tokenizer.json")
tokenizer.save(tokenizer_path)
# 读取
tokenizer = Tokenizer.from_file(tokenizer_path)
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



# ============== 基于 wikitext 语料，从头开始训练 BERT 模型 ==============
def __Training_bert_on_wikitext():
    pass

# %% ------------------- 封装 Tokenizer 对象 ------------------
# 1. 加载上面训练好的 tokenizer
tokenizer_path = os.path.join(LOCAL_DATA_PATH, 'huggingface', f"{split}_tokenizer.json")
raw_tokenizer = Tokenizer.from_file(tokenizer_path)
# 注意，这个 tokenizer 是 tokenizers包中的 Tokenizer 类对象，不能直接被transformer直接使用
print(type(raw_tokenizer))
# 封装成 PreTrainedTokenizerFast对象，这里还需要设置特殊token的映射关系
custom_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    cls_token='[CLS]', unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', mask_token='[MASK]'
)
# print(custom_tokenizer.is_fast)
# print(custom_tokenizer.unk_token, custom_tokenizer.cls_token, custom_tokenizer.sep_token, custom_tokenizer.pad_token, custom_tokenizer.mask_token)
# print(custom_tokenizer.all_special_tokens)
# print(custom_tokenizer.all_special_ids)

# 对比一下官方的BERT分词器
# model_path = os.path.join(CWD, 'bert-pretrained-models', 'bert-base-uncased')
# bert_tokenizer = BertTokenizerFast.from_pretrained(model_path)
# print(type(bert_tokenizer))
# print(bert_tokenizer.unk_token, bert_tokenizer.cls_token, bert_tokenizer.sep_token, bert_tokenizer.pad_token, bert_tokenizer.mask_token)
# print(bert_tokenizer.all_special_tokens)
# print(bert_tokenizer.all_special_ids)

# sentence = ["After stealing money from the bank vault", "the bank robber was seen fishing on the Mississippi river bank."]
# custom_encoding = custom_tokenizer(*sentence, padding=True)
# bert_encoding = bert_tokenizer(*sentence, padding=True)
# print(custom_encoding)
# print(bert_encoding)


# %% ------------------- 处理数据 ------------------------
# 这里处理成MLM的形式
# 1. 使用分词器进行分词
def tokenize(examples):
    # 去掉空白行的样本
    text_list = [text for text in examples['text'] if len(text) > 1]
    # 分词处理
    text_tokens = custom_tokenizer(text_list)
    return text_tokens

wikitext_tokenized = wikitext_train.map(tokenize, batched=True, remove_columns=['text'], load_from_cache_file=True)
print(wikitext_tokenized)

# 2. 分成大小一致的chunk
chunk_size = 128
def generate_chunks(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # 计算此 batch 的样本拼接后的总长度，list(examples.keys())[0] 就是 input_ids 这个key
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 对于最后一个chunk，它的长度可能小于 chunk_size，这里直接丢弃
    total_length = (total_length // chunk_size) * chunk_size
    # 对拼接后的 token 序列按照 chunk_size 进行分割
    # k 依次是 input_ids, attention_mask, word_ids —— 也就是依然是这 3 个特征
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column —— 这个 label 后面会作为 mask 词的预测标签
    result["labels"] = result["input_ids"].copy()
    return result

wikitext_chunks = wikitext_tokenized.map(generate_chunks, batched=True, load_from_cache_file=True)

# 3. 设置训练数据生成器
wikitext_collator = DataCollatorForWholeWordMask(tokenizer=custom_tokenizer, mlm_probability=0.15)
wikitext_dataloader = DataLoader(dataset=wikitext_chunks, batch_size=24, shuffle=True, collate_fn=wikitext_collator)

# %% ------------------- 初始化模型 -----------------------
# 初始化一个新的 BERT 模型
custom_config = BertConfig(
    vocab_size=custom_tokenizer.vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=8,
    intermediate_size=2048,
)
# custom_bert = BertForPreTraining(custom_config)
custom_bert = BertForMaskedLM(custom_config)

# %% ------------------ 训练模型 -----------------------
# GPU 设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
custom_bert.cuda(device)
# print(device)
# 定义优化器
optimizer = optim.SGD(params=custom_bert.parameters(), lr=0.1)
# 进行训练
num_train_epochs = 5
loss = 0
print(f"training on device: {device}")
start_time = time()
for epoch in range(1, num_train_epochs+1):
    custom_bert.train()
    for batch_num, batch in enumerate(wikitext_dataloader, start=1):
        input_ids, labels = batch['input_ids'], batch['labels']
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = custom_bert(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        if batch_num % 100 == 0:
            print(f"training loss at batch '{batch_num}' of epoch '{epoch}' is: {loss}")
        loss.backward()
        optimizer.step()
    print(f"training loss finally at epoch '{epoch}' is : {loss}")
end_time = time()
total_time = end_time - start_time
print(f"total training time is : {total_time}")
