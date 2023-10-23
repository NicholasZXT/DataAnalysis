import os
import json
from typing import List
from collections import Counter
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, default_collate
from datasets import GeneratorBasedBuilder, Version, BuilderConfig, DatasetInfo, Features, Value, DownloadManager, \
    SplitGenerator, Split

DATA_PATH = r"C:\Users\Drivi\Documents\技术书籍\datasets\瑞金MMC知识图谱构建\0521_new_format"
DATA_PROC_PATH = r"C:\Users\Drivi\Documents\技术书籍\datasets\瑞金MMC知识图谱构建\process"
# DATA_PATH = r"D:\Project-Workspace\Python-Projects\DataAnalysis\local-datasets\MMC知识图谱构建\0521_new_format"
# DATA_PROC_PATH = r"D:\Project-Workspace\Python-Projects\DataAnalysis\local-datasets\MMC知识图谱构建\process"

# MMC 数据集实体类型
# 数据集中 Test_Items 写成了 Test_items
ENTITY_TYPES = ['Disease', 'Class', 'Reason', 'Pathogenesis', 'Symptom', 'Test', 'Test_items', 'Test_Value', 'Drug',
                 'Frequency', 'Amount', 'Method', 'Treatment', 'Operation', 'ADE', 'Anatomy', 'Level', 'Duration']
# MMC 数据集实体关系类型
ENTITY_RELATIONS = []


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


class MmcVocabulary:
    """
    构建MMC语料库的词表，实现 字到词表索引 的 双向映射
    """
    def __init__(self):
        self._word2id = dict()
        # 两个特殊的字符
        self._PAD_TOKEN = 'PAD'
        self._UNK_TOKEN = 'UNK'
        self._word2id[self._PAD_TOKEN] = 0
        self._word2id[self._UNK_TOKEN] = 1
        self._id2word = ['PAD', 'UNK']

    @property
    def PAD_TOKEN(self):
        return self._PAD_TOKEN

    @property
    def UNK_TOKEN(self):
        return self._UNK_TOKEN

    @property
    def vocab_size(self):
        return len(self)

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


class EntityTagMap:
    """
    根据实体类型列表，构建 BIOES 的实体标记 tag，并进行对应的转换
    """
    marks = {'begin': 'B-', 'middle': 'I-', 'end': 'E-', 'single': 'S-'}

    def __init__(self, entity_types: List[str]):
        self.entity_types = set(entity_types)
        # 非实体对应的tag
        self._id2tag = ['O']
        self._tag2id = {'O': 0}

    @property
    def tag_num(self):
        return len(self)

    def __len__(self):
        return len(self._id2tag)

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
        比如传入为 [LOC, PER]，构建一个 BIOES 标注序列的实体索引字典为：
        {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'E-LOC': 3, 'S-LOC': 4, 'B-PER': 5, 'I-PER': 6', 'E-PER': 7, 'S-PER': 8}
        """
        entity_map = cls(entity_types=entity_types)
        # marks = ['B-', 'I-', 'E-', 'S-']
        marks = entity_map.marks.values()
        index = 1  # 0 留给了初始化时的非实体tag
        for entity in entity_types:
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
        根据传入的实体类型，和实体的标记位置 {begin, middle, end, single}，返回实体类型的 tag 和 index
        """
        if entity not in self.entity_types:
            raise ValueError(f"entity type '{entity}' is not valid")
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


# 适用于BiLSTM+CRF的数据集
class MmcDatasetV1(Dataset):
    def __init__(self, data_dir, vocab: MmcVocabulary, entity_types: List[str], debug_limit: int = 0):
        self.vocab = vocab
        self.entity_map = EntityTagMap.generate_entities_bioes_tag(entity_types=entity_types)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"data_dir '{data_dir}' not found")
        # 这个 data_dir 里的内容是经过 sentence_extract 处理的数据，每一个json文件是一个句子
        self.base_dir = data_dir
        self.debug_limit = debug_limit
        if self.debug_limit > 0:
            files = os.listdir(data_dir)
            self._files = files[:self.debug_limit]
        else:
            self._files = os.listdir(data_dir)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        file = self._files[index]
        file_path = os.path.join(self.base_dir, file)
        with open(file_path, mode='r') as f:
            sentence_info = json.load(f)
        # 将句子中的 字 根据词典转成索引
        sent_idx = [self.vocab.word2idx(word) for word in sentence_info['sentence']]
        # 对单个句子进行处理，根据实体信息，将句子转换成实体类型的 BIOES 标记序列
        sent_tags, sent_tags_idx = self.process_sentence_info(sentence_info)
        # 这里并没有对句子进行 padding 或者 truncating 的操作 ------------- KEY
        return sent_idx, sent_tags, sent_tags_idx

    def process_sentence_info(self, sentence_info: dict):
        sent = sentence_info['sentence']
        # 初始构建一个全为 0 的标签序列，以及对应的 tag_index 序列
        sent_tags = ['O' for _ in sent]
        sent_tags_idx = [self.entity_map.tag2idx(tag) for tag in sent_tags]
        sent_entities = sentence_info['entities']
        if len(sent_entities) == 0:
            return sent_tags, sent_tags_idx
        # sent_entities_df = pd.DataFrame(sent_entities)
        # start_idx 是实体开始的索引（包含），end_idx 是实体结束的索引（不包含）
        # 会出现实体嵌套、实体交叉的情况  --------------------- KEY
        # 对所有实体按照 start_idx 升序，end_idx 降序排列，然后检查前后的实体是否有交叉、嵌套的情况
        sent_entities = sorted(sent_entities, key=lambda item: (item['start_idx'], -item['end_idx']))
        sent_entities[0]['valid'] = True  # 第一个实体标记为有效
        self.label_sent_with_valid_entity(sent_tags, sent_entities[0], self.entity_map)
        prev_end = sent_entities[0]['end_idx']  # 前一个有效实体的结束索引
        # 从第二个实体开始遍历检查
        for i in range(1, len(sent_entities)):
            pre_entity = sent_entities[i - 1]
            cur_entity = sent_entities[i]
            # 当前实体的 start 在 上一个实体的 end 之前，或者在 上一个有效实体的 end 之前，则丢弃当前实体
            if cur_entity['start_idx'] < pre_entity['end_idx'] or cur_entity['start_idx'] < prev_end:
                cur_entity['valid'] = False
            else:
                # 当前实体没有和前面的实体交叉或者重叠，有效
                cur_entity['valid'] = True
                # 对 sent_tags 进行标记
                self.label_sent_with_valid_entity(sent_tags, cur_entity, self.entity_map)
                prev_end = cur_entity['end_idx']
        sent_tags_idx = [self.entity_map.tag2idx(tag) for tag in sent_tags]
        return sent_tags, sent_tags_idx

    @staticmethod
    def label_sent_with_valid_entity(sent_tags: List[str], entity_item: dict, entity_map: EntityTagMap):
        """根据有效的实体 entity_item，对 sent_tags 进行标记"""
        entity = entity_item['entity_type']   # 实体类型
        start_idx = entity_item['start_idx']  # 实体包含开始的索引
        end_idx = entity_item['end_idx']      # 但不包含结束的索引
        if start_idx >= end_idx:
            pass
        elif start_idx == end_idx - 1:
            sent_tags[start_idx] = entity_map.get_entity_tag(entity, 'single')
        else:
            sent_tags[start_idx] = entity_map.get_entity_tag(entity, 'begin')
            for i in range(start_idx+1, end_idx):
                sent_tags[i] = entity_map.get_entity_tag(entity, 'middle')
            sent_tags[end_idx-1] = entity_map.get_entity_tag(entity, 'end')


class MmcDatasetV1CollateFun:
    def __init__(self, max_len: int, entity_map: EntityTagMap, vocab: MmcVocabulary):
        self.max_len = max_len
        self.entity_map = entity_map
        self.vocab = vocab

    def __call__(self, batch):
        sent_idx_batch = []
        sent_tags_batch = []
        sent_tags_idx_batch = []
        for sent_idx, sent_tags, sent_tags_idx in batch:
            diff = len(sent_tags) - self.max_len
            if diff < 0:
                # 需要在末尾补齐
                padding_words = [self.vocab.word2idx(self.vocab.PAD_TOKEN) for _ in range(abs(diff))]
                padding_tags = ['O' for _ in range(abs(diff))]
                padding_tags_idx = [self.entity_map.tag2idx(tag) for tag in padding_tags]
                sent_idx.extend(padding_words)
                sent_tags.extend(padding_tags)
                sent_tags_idx.extend(padding_tags_idx)
            elif diff > 0:
                # 需要截断
                sent_idx = sent_idx[:-diff]
                sent_tags = sent_tags[:-diff]
                sent_tags_idx = sent_tags_idx[:-diff]
            else:
                pass
            sent_idx_batch.append(sent_idx)
            sent_tags_batch.append(sent_tags)
            sent_tags_idx_batch.append(sent_tags_idx)
        # 这里不能使用默认的这个合并方法，它会将一个batch中对应位置的标签合并到一起，最终形成 max_len 个数据，不是我想要的形式
        # res = default_collate(batch=batch)
        sent_idx_batch_tensor = torch.tensor(sent_idx_batch)
        sent_tags_idx_batch_tensor = torch.tensor(sent_tags_idx_batch)
        return sent_idx_batch_tensor, sent_tags_batch, sent_tags_idx_batch_tensor


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
    vocab = MmcVocabulary.generate_vocabulary(DATA_PATH)
    # vocab.idx2word(0)
    # vocab.idx2word(1)
    # vocab.idx2word(2)
    # vocab.word2idx('UNK')
    # vocab.word2idx('PAD')
    # vocab.word2idx('糖')

    # 将原始的以 doc 为文件的数据，抽取成以 sentence 为文件的数据
    # sentence_extract(DATA_PATH, DATA_PROC_PATH)

    # entities = ['LOC', 'PERSON']
    # entity_map = EntityTagMap.generate_entities_bioes_tag(entities)
    entity_map = EntityTagMap.generate_entities_bioes_tag(ENTITY_TYPES)
    # print(entity_map)

    mmc_ds = MmcDatasetV1(data_dir=DATA_PROC_PATH, vocab=vocab, entity_types=ENTITY_TYPES, debug_limit=20)
    collate_fn = MmcDatasetV1CollateFun(max_len=64, entity_map=entity_map, vocab=vocab)
    mmc_loader = DataLoader(dataset=mmc_ds, batch_size=4, collate_fn=collate_fn)
    for sent_idx_tensor, sent_tags, sent_tags_idx_tensor in mmc_loader:
        print(sent_idx_tensor)
        print(sent_tags)
        print(sent_tags_idx_tensor)



