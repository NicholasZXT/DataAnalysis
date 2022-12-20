# 用于练习 pytorch本身的transformer 和 huggingface 的 transformer
import os
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, \
    TransformerDecoderLayer, Transformer
from transformers import BertTokenizer, DistilBertTokenizer, BertForSequenceClassification, \
    DistilBertForSequenceClassification, AdamW


# %% ----------- pytorch Transformer-MultiheadAttention ---------------
def __Pytorch_MultiHeadAttention():
    pass

embed_dim, num_heads = 512, 8
batch_size, target_len, source_len = 5, 10, 20

query = torch.rand(batch_size, target_len, embed_dim)
key = torch.rand(batch_size, source_len, embed_dim)
value = torch.rand(batch_size, source_len, embed_dim)

multihead_attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                    dropout=0, bias=False, add_bias_kv=False, batch_first=True)
attn_output, attn_output_weights = multihead_attn(query, key, value)
print(attn_output.shape)
print(attn_output_weights.shape)


# %% ------------- pytorch Transformer-------------------------------------
def __Pytorch_Transformer():
    pass

# Encoder层，需要设置的参数有：
# 输入特征数 in_features=56, nhead, dim_feedforward是指self-attention后的前馈网络的中间层，
# Encoder的不会改变输入样本的特征数量，也就是说，输入的in_features是多少，输出的就是多少
# in_features = 56
# encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=4, dim_feedforward=24)
# # 输入数据，batch_size=10, seq_length = 32, 特征数为in_features
# src = torch.rand(10, 32, in_features)
# out = encoder_layer(src)
# src.shape
# out.shape
#
#
# X.shape
# input_size = 96
# hidden_size = 48
# seq_length = 95
# encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, dim_feedforward=hidden_size, nhead=4)
# out = encoder_layer(X.transpose(1, 0))
# out.shape



# %% ------------- pytorch transformer 的IMDB练习 ------------------
def __PyTorch_Transformer_IMDB():
    pass

input_size = 96
hidden_size = 48
seq_length = 95
class ImdbTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_size, dim_feedforward=hidden_size, nhead=4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=seq_length*input_size, out_features=2, bias=True)

    def forward(self, X):
        X_transpose = X.transpose(1,0)
        trans_out = self.transformer(X_transpose)
        trans_flatten = self.flatten(trans_out.transpose(1,0))
        y_pred = self.linear(trans_flatten)
        return y_pred


# 初始化模型
imdb_trans = ImdbTransformer(input_size=input_size, hidden_size=hidden_size, seq_length=seq_length)
# 测试输出
# y_pred = imdb_trans(X)
# y_pred.shape
# 定义损失函数
crossEnt = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.SGD(params=imdb_trans.parameters(), lr=0.1)
# 测试
# out = imdb_trans(X)
# loss = crossEnt(out, y.long())

# 开始迭代训练
for epoch in range(1, 5):
    for batch_id, (X, y) in enumerate(imdb_train_loader):
        y_pred = imdb_trans(X)
        optimizer.zero_grad()
        loss = crossEnt(y_pred, y.long())
        loss.backward()
        optimizer.step()
        y_pred_new = imdb_trans(X)
        train_loss = crossEnt(y_pred_new, y.long())
        print("epoch {:2}, batch_id {:2}, train_loss: {}".format(epoch, batch_id, train_loss))


# --------- 损失函数相关 -------------------

# negative log likelihood loss
nlloss = nn.NLLLoss(reduction='none')
y_true = torch.tensor([1, 0, 0], dtype=torch.long)
y_pred = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.float, requires_grad=True)
y_true
y_pred
nlloss(input=y_pred, target=y_true)


embed_dim, num_heads = 512, 8
E, S, L, N = embed_dim, 20, 10, 5
query = torch.rand(L, N, E)
key = torch.rand(S, N, E)
value = torch.rand(S, N, E)
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(query, key, value)

embed_dim, num_heads = 512, 8
E, S, L, N = embed_dim, 20, 10, 5
query = torch.rand(L, N, E)
key = torch.rand(S, N, 256)
value = torch.rand(S, N, 256)
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, kdim=256, vdim=256)
attn_output, attn_output_weights = multihead_attn(query, key, value, )


# %% ----------------- huggingface transformer ------------------
def __Huggingface_Transformer():
    pass

# --------------------- 在IMDB数据集上 fine-tune BERT模型-----------------------------------------------------
def __BERT_fine_tune():
    pass
# 加载预训练模型 和 Tokenizer
# model_path = r"BERT-source-codes\bert-pre-trained-models\bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
# model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
model_path = r"BERT-source-codes\bert-pre-trained-models\distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
model = DistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)

# 在GPU上进行训练
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
model.to(device)


# 设置读取的数据集大小
train_size, train_batch_size = 2000, 20
test_size, test_batch_size = 100, 20


#读取数据
def read_imdb_split(split_dir, limit=None):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for i, text_file in enumerate((split_dir/label_dir).iterdir()):
            if limit and i >= limit:
                break
            texts.append(text_file.read_text(encoding='utf-8'))
            labels.append(0 if label_dir is "neg" else 1)
    return texts, labels


data_train_dir = r"./datasets/aclImdb/train"
data_test_dir = r"./datasets/aclImdb/test"
# Path(data_train_dir)
# os.path.exists(data_train_dir)
# train_texts, train_labels = read_imdb_split(data_train_dir)
# test_texts, test_labels = read_imdb_split(data_test_dir)
train_texts, train_labels = read_imdb_split(data_train_dir, limit=train_size)
test_texts, test_labels = read_imdb_split(data_test_dir, limit=test_size)
# len(train_texts)
# 使用sklearn读取数据
# text_train = load_files(data_train_dir)
# t = tokenizer(train_texts[0])

# 切分训练集和验证集
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
# len(train_texts)
# len(val_texts)

# 对原始文本进行分词
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
# val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# 构建 Dataset
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
# val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
# train_dataset[0]
# test_dataset[0]["input_ids"].shape
# 获取测试集，这里只取了一部分
batch_test_iter = iter(test_loader)
batch_test = next(batch_test_iter)
test_input_ids, test_attention_mask, test_labels = batch_test["input_ids"].to(device), batch_test['attention_mask'].to(device), batch_test['labels'].to(device)


# 验证一下输出
# batch_iter = iter(train_loader)
# batch = next(batch_iter)
# len(batch["input_ids"])
# input_ids, attention_mask, labels = batch["input_ids"].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
# outputs = model(input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True, output_hidden_states=True)
# loss = outputs.loss  # 这里的 Loss 是交叉熵损失
# logits = outputs.logits  # logits 是输出的取各个类别的原始值，没有经过 softmax + log 变换
# # hidden_states = outputs.hidden_states
# # attentions = outputs.attentions
# # 手动计算 交叉熵，可以看出，两者是一样的.
# crossEntropy = nn.CrossEntropyLoss()
# loss_manual = crossEntropy(logits, labels)
# print(loss, loss_manual)
# # 查看隐藏层状态和对应attention
# # hidden_states[0].shape
# # attentions[0].shape
# # 将 logits 转换成预测的概率
# softmax = nn.Softmax(dim=1)
# logits_soft = softmax(logits)
# # 再转换成预测结果，也就是获取每一行中概率最大处的index
# labels_pred = torch.argmax(logits_soft, dim=1)
# print(labels_pred[:5], labels[:5])
# # 计算分类错误率
# diff = torch.abs(labels_pred - labels).numpy()
# diff.sum()/diff.size


# 训练前的配置
optim = AdamW(model.parameters(), lr=5e-5)
softmax = nn.Softmax(dim=1)
# 开始训练
start_time = time.time()
for epoch in range(5):
    model.train()
    for batch_id, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
        if batch_id % 5 == 0:
            # 计算分类错误率
            logits_soft = softmax(outputs.logits)
            labels_pred = torch.argmax(logits_soft, dim=1)
            diff = torch.abs(labels_pred - labels).cpu().numpy()
            error = diff.sum() / diff.size
            if error >= 1.0 :
                print("diff.sum(): ", diff.sum(), "; diff.size: ", diff.size)
            print("epoch {:2} - training loss on batch {:2} is: {:.4f}, error rate is {:.2f}%.".format(epoch+1, batch_id, loss, error*100))
    model.eval()
    test_outputs = model(test_input_ids, attention_mask=test_attention_mask, labels=test_labels)
    test_loss = test_outputs.loss
    test_labels_pred = torch.argmax(softmax(test_outputs.logits), dim=1)
    test_diff = torch.abs(test_labels_pred - test_labels).cpu().numpy()
    test_error = test_diff.sum()/test_diff.size
    print("\nepoch {:2} - testing loss is: {:.4f}, error rate is {:2f}%".format(epoch+1, test_loss, test_error*100))
    print("---------------------------------------")
end_time = time.time()
print("time cost is : {:.2f}.".format(end_time-start_time))

# 以下是 distill-bert 耗时
# cpu训练，
# train_size, train_batch_size = 100, 20
# test_size, test_batch_size = 100, 20
# 175.57
# GPU训练
# train_size, train_batch_size = 200, 20
# test_size, test_batch_size = 100, 20
# 11.19
# GPU训练
# train_size, train_batch_size = 2000, 20  这个样本量得到的模型在测试集上的效果已经很好了，错误率接近于0
# test_size, test_batch_size = 100, 20
# 181.83

# RTX 2060 的 6G 根本带不动 BERT-source-codes，连 distill-bert 都很困难
# distill-bert 模型时，只计算一个batch时，最大的 batch_size 只能是 60, 即使是 65 都会报OOM；
# 如果要循环迭代，最大的 batch_size 只能是20（训练和测试都是20）,25都会报OOM.


# TODO
# %% --------------- 手动训练一个BERT模型-----------------
def __BERT_Training():
    pass
# 使用的是 世界语 esperanto 的语料库
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer # 这个tokenizer包也是huggingfacing开源的，配合transformer使用
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

# t = Path("./datasets/Esperanto-Corpus/").glob("**/*.txt")
# list(t)
paths = [str(x) for x in Path("./datasets/Esperanto-Corpus/").glob("**/*.txt")]
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=paths, vocab_size=52000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
tokenizer.save_model(r"./datasets/Esperanto-Corpus/", "esperberto")

tokenizer = ByteLevelBPETokenizer(r"./datasets/Esperanto-Corpus/esperberto-vocab.json",
                                  r"./datasets/Esperanto-Corpus/esperberto-merges.txt",
                                  )
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
print(tokenizer.encode("Mi estas Julien."))
tokenizer.__class__

from torch.utils.data import Dataset

class EsperantoDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = ByteLevelBPETokenizer(
            r"./datasets/Esperanto-Corpus/esperberto-vocab.json",
            r"./datasets/Esperanto-Corpus/esperberto-merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        src_files = Path("./datasets/Esperanto-Corpus/").glob("**/*.txt") if evaluate else Path("./datasets/Esperanto-Corpus/").glob("**/*.txt")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


