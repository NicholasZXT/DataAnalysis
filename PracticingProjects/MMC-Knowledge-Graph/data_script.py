import os
import json
from typing import List
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, default_collate
from datasets import GeneratorBasedBuilder, Version, BuilderConfig, DatasetInfo, Features, Value, DownloadManager, \
    SplitGenerator, Split

DATA_PATH = r"C:\Users\Drivi\Documents\技术书籍\datasets\瑞金MMC知识图谱构建\0521_new_format"
OUT_PATH = r"C:\Users\Drivi\Documents\技术书籍\datasets\瑞金MMC知识图谱构建\process"
# MMC 数据集实体类型
ENTITIES_TYPE = ['Disease', 'Class', 'Reason', 'Pathogenesis', 'Symptom', 'Test', 'Test_Items', 'Test_Value', 'Drug',
                 'Frequency', 'Amount', 'Method', 'Treatment', 'Operation', 'ADE', 'Anatomy', 'Level', 'Duration']
# MMC 数据集实体关系类型
ENTITIES_RELATIONS = []


def sentence_extract(data_path, out_path):
    """
    将训练语料所有 doc中 每个 paragraph 的 sentence 抽取出来单独存放到一个 json 文件里
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"not found data path: {data_path}")
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"not found output path: {out_path}")
    files = os.listdir(data_path)
    for file in files:
        file_path = os.path.join(data_path, file)
        print(f"processing file: {file_path}")
        with open(file_path, mode='r', encoding='utf-8') as f:
            doc = json.load(f)
            doc_id = doc['doc_id']
            for para in doc['paragraphs']:
                paragraph_id = para['paragraph_id']
                for sentence in para['sentences']:
                    sentence_id = sentence['sentence_id']
                    sentence['doc_id'] = doc_id
                    sentence['paragraph_id'] = paragraph_id
                    with open(os.path.join(out_path, f"{doc_id}-{paragraph_id}-{sentence_id}.json"),
                              mode='w') as fout:
                        # fout.write(json.dumps(sentence))
                        fout.write(json.dumps(sentence, ensure_ascii=False))


class EntityTagMap:
    """
    根据实体类型列表，构建 BIOS 的实体标记 tag，并进行对应的转换
    """
    marks = {'begin': 'B-', 'middle': 'I-', 'end': 'E-', 'single': 'S-'}

    def __init__(self, entity_types):
        self.entity_types = entity_types
        # 非实体对应的tag
        self._id2tag = ['O']
        self._tag2id = {'O': 0}

    def update(self, idx, tag):
        if idx < len(self._id2tag):
            self._id2tag[idx] = tag
        elif idx == len(self._id2tag):
            self._id2tag.append(tag)
        else:
            raise IndexError(f"new index out of range, please add tag as order.")
        self._tag2id[tag] = idx

    @classmethod
    def generate_entities_bioes_tag(cls, entity_types: List[str]):
        """
        根据传入的实体类别，构建 BIOES 标注方式的实体索引字典。
        比如传入为 [LOC, PER]，构建一个BIO标注序列的实体索引字典为：
        {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'E-LOC': 3, 'S-LOC': 4, 'B-PER': 5, 'I-PER': 6', 'E-PER': 7, 'S-PER': 8}
        """
        entity_map = cls(entity_types=entity_types)
        # marks = ['B-', 'I-', 'E-', 'S-']
        marks = entity_map.marks.values()
        index = 1  # 0 留给了初始化时的非实体tag
        for entity in entities:
            for mark in marks:
                tag = mark + entity
                entity_map.update(index, tag)
                index += 1
        return entity_map

    def idx2tag(self, index):
        return self._id2tag[index]

    def tag2idx(self, tag):
        return self._tag2id[tag]

    def get_entity_tag(self, entity: str, category: str):
        """
        根据传入的实体，和实体的标记位置 {begin, middle, end, single}，返回实体的 tag 和 index
        """
        if entity not in self.entity_types:
            raise ValueError(f"entity '{entity}' is not valid")
        if category not in self.marks:
            raise ValueError(f"cagetory value must in {self.marks.keys()}")
        mark = self.marks[category]
        entity_tag = mark + entity
        return entity_tag
        # entity_tag_idx = self.tag2idx(entity_tag)
        # return entity_tag, entity_tag_idx

    def __repr__(self):
        id2tags = {idx: self._id2tag[idx] for idx in range(len(self._id2tag))}
        out = f"""{{\n'id2tags': {str(id2tags)}, \n'tags2id': {str(self._tag2id)}\n}}"""
        return str(out)



class MmcVocabulary:
    """
    构建MMC语料库的词表，实现 字到词表索引 的 双向映射
    """
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

    def word2idx(self, word: str):
        # 没找到就返回 UNK 的 id
        return self._word2id.get(word, 1)

    def idx2word(self, idx):
        return self._id2word[idx]
        # if 0 <= idx < len(self._id2word):
        #     return self._id2word[idx]
        # else:
        #     raise IndexError(f"index out of range: {idx}")

    @classmethod
    def generate_vocabulary(cls, data_path: str):
        """
        遍历语料数据集，构建中文的字符集合
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"not found data path: {data_path}")
        files = os.listdir(data_path)
        counter = Counter()
        vocab = cls()
        for file in files:
            file_path = os.path.join(data_path, file)
            print(f"processing file: {file_path}")
            with open(file_path, mode='r', encoding='utf-8') as f:
                doc = json.load(f)
                for para in doc['paragraphs']:
                    words = list(para['paragraph'])
                    counter.update(words)
        vocab.init_from_counter(counter=counter, min_freq=2)
        return vocab


# 适用于BiLSTM+CRF的数据集
class MmcDatasetV1(Dataset):
    def __init__(self, data_dir, vocab: MmcVocabulary, entites_type):
        self.vocab = vocab
        self.tag2idx, self.idx2tag = EntityTagMap.generate_entities_bioes_tag(entities=entites_type)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"data_dir '{data_dir}' not found")
        self.base_dir = data_dir
        self._files = os.listdir(data_dir)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        file = self._files[index]
        file_path = os.path.join(self.base_dir, file)




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
    # os.path.exists(DATA_PATH)
    # files = os.listdir(DATA_PATH)
    # train_val, test = train_test_split(files, train_size=0.8, random_state=32, shuffle=True)
    # train, validation = train_test_split(files, train_size=0.8, random_state=32, shuffle=True)
    # data_jsons = []
    # for file in test:
    #     file = '1.json'
    #     file_path = os.path.join(DATA_PATH, file)
    #     with open(file_path, mode='r', encoding='utf-8') as f:
    #         doc = json.load(f)
    #         data_jsons.append(doc)

    vocab = MmcVocabulary.generate_vocabulary(DATA_PATH)
    vocab.idx2word(0)
    vocab.idx2word(1)
    vocab.idx2word(2)
    vocab.word2idx('UNK')
    vocab.word2idx('PAD')
    vocab.word2idx('糖')

    entities = ['LOC', 'PERSON']
    # entity_map = EntityTagMap.generate_entities_bioes_tag(entities)
    entity_map = EntityTagMap.generate_entities_bioes_tag(ENTITIES_TYPE)
    # print(entity_map)

    sentence_extract(DATA_PATH, OUT_PATH)

    file = os.path.join(OUT_PATH, '1-0-0.json')
    with open(file) as f:
        sample = json.load(f)
    sent = sample['sentence']
    sent_entities = sample['entities']
    sent_tags = ['O' for _ in sent]
    for item in entities:
        start_idx = item['star_idx']
        end_idx = item['end_idx']
        entity = item['entity_type']
        if start_idx == end_idx:
            sent_tags[start_idx] = entity_map.get_entity_tag(entity, 'single')
        else:
            for i in range(start=start_idx, stop=end_idx+1):
                pass

