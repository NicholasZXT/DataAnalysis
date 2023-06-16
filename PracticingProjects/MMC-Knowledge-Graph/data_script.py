import os
import json
from typing import List
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from datasets import GeneratorBasedBuilder, Version, BuilderConfig, DatasetInfo, Features, Value, DownloadManager, \
    SplitGenerator, Split

DATA_PATH = r"C:\Users\Drivi\Documents\技术书籍\datasets\瑞金MMC知识图谱构建\0521_new_format"


class MmcDatasetV1(Dataset):
    def __init__(self):
        pass

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


def generate_entities_bioes_map(entities: List[str]):
    """
    根据传入的实体类别，构建 BIOES 标注方式的实体索引字典。
    比如传入为 [LOC, PER]，构建一个BIO标注序列的实体索引字典为：
    {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'E-LOC': 3, 'S-LOC': 4, 'B-PER': 5, 'I-PER': 6', 'E-PER': 7, 'S-PER': 8}
    """
    index = 0
    marks = ['B-', 'I-', 'E-', 'S-']
    entities_to_index = {'O': index}
    index_to_entities = {index: 'O'}
    for entity in entities:
        for mark in marks:
            item = mark+entity
            entities_to_index[item] = index
            index_to_entities[index] = item
            index += 1
    return entities_to_index, index_to_entities


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
            data = json.load(f)
            data_jsons.append(data)

