import os
import json
from typing import List
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from datasets import GeneratorBasedBuilder, Version, BuilderConfig, DatasetInfo, Features, Value, DownloadManager, \
    SplitGenerator, Split

DATA_PATH = r"C:\Users\Drivi\Documents\技术书籍\datasets\瑞金MMC知识图谱构建\0521_new_format"
# MMC 数据集实体类型
ENTITIES_TYPE = ['Disease', 'Class', 'Reason', 'Pathogenesis', 'Symptom', 'Test', 'Test_Items', 'Test_Value', 'Drug',
            'Frequency', 'Amount', 'Method', 'Treatment', 'Operation', 'ADE', 'Anatomy', 'Level', 'Duration']
# MMC 数据集实体关系类型
ENTITIES_RELATIONS = []


class MmcVocabulary:
    def __init__(self):
        self._word2id = dict()
        # 两个特殊的字符
        self._word2id['PAD'] = 0
        self._word2id['UNK'] = 1
        self._id2word = ['PAD', 'UNK']

    def __len__(self):
        return len(self._id2word)

    def init_from_counter(self, counter: Counter, min_freq: int = 0):
        vocab = [word for word, cnt in counter.items() if cnt >= min_freq]
        for idx, word in enumerate(vocab, start=2):
            self._word2id[word] = idx
            self._id2word.append(word)

    def word2id(self, word: str):
        # 没找到就返回 UNK 的 id
        return self._word2id.get(word, 1)

    def id2word(self, idx):
        return self._word2id[idx]
        # if 0 <= idx < len(self._id2word):
        #     return self._word2id[idx]
        # else:
        #     raise IndexError(f"index out of range: {idx}")

    @classmethod
    def generate_vocabulary(cls, data_path: str):
        """
        遍历语料数据集，构建中文的字符集合
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"not found data path: {data_path}")
        files = os.listdir(DATA_PATH)
        counter = Counter()
        vocab = cls()
        for file in files:
            file_path = os.path.join(DATA_PATH, file)
            with open(file_path, mode='r', encoding='utf-8') as f:
                doc = json.load(f)
                for para in doc['paragraphs']:
                    words = list(para['paragraph'])
                    counter.update(words)
        vocab.init_from_counter(counter=counter, min_freq=2)
        return vocab

    @staticmethod
    def generate_entities_bioes_tag(entities: List[str]):
        """
        根据传入的实体类别，构建 BIOES 标注方式的实体索引字典。
        比如传入为 [LOC, PER]，构建一个BIO标注序列的实体索引字典为：
        {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'E-LOC': 3, 'S-LOC': 4, 'B-PER': 5, 'I-PER': 6', 'E-PER': 7, 'S-PER': 8}
        """
        index = 0
        marks = ['B-', 'I-', 'E-', 'S-']
        tag_to_index = {'O': index}
        index_to_tag = {index: 'O'}
        for entity in entities:
            for mark in marks:
                item = mark + entity
                tag_to_index[item] = index
                index_to_tag[index] = item
                index += 1
        return tag_to_index, index_to_tag


# 适用于BiLSTM+CRF的数据集
class MmcDatasetV1(Dataset):
    def __init__(self, data_dir):
        self.vocab = MmcVocabulary.generate_vocabulary(DATA_PATH)
        self.tag2idx, self.idx2tag = MmcVocabulary.generate_entities_bioes_tag(entities=ENTITIES_TYPE)

    def __getitem__(self, item):
        pass

    def __len__(self):
        return 0


# 适用于HF的数据集类
class MmcDataset(GeneratorBasedBuilder):
    VERSION = Version('1.0.0')
    BUILDER_CONFIGS = [BuilderConfig(name="MMC-dataset")]
    DEFAULT_CONFIG_NAME = "MMC-dataset"
    _DESCRIPTION = "阿里云天池大赛MMC知识图谱构建数据集"
    _URLS = {
        "all": DATA_PATH
    }

    def __init__(self):
        super().__init__()
        self._random_state = 32
        data_dir = self._URLS['all']
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"not found data dir: {data_dir}")
        files = os.listdir(data_dir)
        train_val, test = train_test_split(files, train_size=0.8, random_state=32, shuffle=True)
        train, validation = train_test_split(files, train_size=0.8, random_state=32, shuffle=True)
        self._files_train = train
        self._files_test = test
        self._files_validation = validation

    def _info(self) -> DatasetInfo:
        features = {
            "tokens": Value(dtype='string'),
            'entities': [
                {'end': Value(dtype='int64', id=None),
                 'start': Value(dtype='int64', id=None),
                 'type': Value(dtype='string', id=None)}
            ],
            'ids': Value(dtype='string', id=None),
        }
        info = DatasetInfo(
            description=self._DESCRIPTION,
            features=Features(features)
        )
        return info

    def _split_generators(self, dl_manager: DownloadManager):
        data_dir = dl_manager.download_and_extract(self._URLS['all'])
        splits = [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={'split': 'train', 'filepath': data_dir, 'files': self._files_train}
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={'split': 'test', 'filepath': data_dir, 'files': self._files_test}
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={'split': 'validation', 'filepath': data_dir, 'files': self._files_validation}
            ),
        ]
        return splits

    def _generate_examples(self, **kwargs):
        pass



if __name__ == '__main__':
    # entities = ['LOC', 'PERSON', 'TIME']
    # ent2idx, idx2ent = generate_entities_bioes_map(entities)

    os.path.exists(DATA_PATH)
    files = os.listdir(DATA_PATH)
    train_val, test = train_test_split(files, train_size=0.8, random_state=32, shuffle=True)
    train, validation = train_test_split(files, train_size=0.8, random_state=32, shuffle=True)
    data_jsons = []
    for file in test:
        file = '1.json'
        file_path = os.path.join(DATA_PATH, file)
        with open(file_path, mode='r', encoding='utf-8') as f:
            doc = json.load(f)
            data_jsons.append(doc)

    vocab = MmcVocabulary.generate_vocabulary(DATA_PATH)

