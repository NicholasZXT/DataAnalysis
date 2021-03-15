# 用于练习 huggingface 的 transformer
import os
import torch
from pathlib import Path
from sklearn.datasets import load_files

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sentence = "A Titan RTX has 24GB of VRAM"
tokenized_sentence = tokenizer.tokenize(sentence)
print(tokenized_sentence)

encode_sentence_tokens = tokenizer(sentence)
print(encode_sentence_tokens)

decode_sentence = tokenizer.decode(encode_sentence_tokens['input_ids'])
print(decode_sentence)

# --------------------------------------------------------------------------

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text(encoding='utf-8'))
            labels.append(0 if label_dir is "neg" else 1)
    return texts, labels


data_train_dir = r"D:\Projects\DataAnalysis\datasets\aclImdb\train"
data_test_dir = r"D:\Projects\DataAnalysis\datasets\aclImdb\test"
Path(data_train_dir)
os.path.exists(data_train_dir)
train_texts, train_labels = read_imdb_split(data_train_dir)
test_texts, test_labels = read_imdb_split(data_test_dir)
len(train_texts)
# 使用sklearn读取数据
# text_data = load_files(data_train_dir)

from transformers import DistilBertTokenizer
model_path = r"BERT\bert-pre-trained-models\distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)

t = tokenizer(train_texts[0])

class IMDbDataset(torch.utils.data.Dataset):
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



