# 用于练习 huggingface 的 transformer
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, DistilBertTokenizer, BertForSequenceClassification, AdamW


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sentence = "A Titan RTX has 24GB of VRAM"
tokenized_sentence = tokenizer.tokenize(sentence)
print(tokenized_sentence)

encode_sentence_tokens = tokenizer(sentence)
print(encode_sentence_tokens)

decode_sentence = tokenizer.decode(encode_sentence_tokens['input_ids'])
print(decode_sentence)

# ---------------------使用IMDB数据集来 fine-tune BERT模型-----------------------------------------------------


#读取数据
def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text(encoding='utf-8'))
            labels.append(0 if label_dir is "neg" else 1)
    return texts, labels


data_train_dir = r"./datasets/aclImdb/train"
data_test_dir = r"./datasets/aclImdb/test"
Path(data_train_dir)
os.path.exists(data_train_dir)
train_texts, train_labels = read_imdb_split(data_train_dir)
test_texts, test_labels = read_imdb_split(data_test_dir)
len(train_texts)
# 使用sklearn读取数据
# text_train = load_files(data_train_dir)

# 切分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

# 加载 Tokenizer
# model_path = r"BERT\bert-pre-trained-models\distilbert-base-uncased"
# tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
model_path = r"BERT\bert-pre-trained-models\bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
# t = tokenizer(train_texts[0])

# 对原始文本进行分词
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
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
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_dataset[0]


# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
model.to("cpu")
# 验证一下输出
batch_iter = iter(train_loader)
batch = next(batch_iter)
len(batch["input_ids"])
input_ids = batch["input_ids"]
attention_mask = batch['attention_mask']
labels = batch['labels']
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
type(outputs)
outputs.loss

# 在GPU上进行训练
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()
optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()


# TODO
#---------------训练一个BERT模型-----------------
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


