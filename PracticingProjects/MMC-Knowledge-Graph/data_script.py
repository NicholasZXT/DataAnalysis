import os
import datasets
from datasets import GeneratorBasedBuilder, Version, BuilderConfig, DatasetInfo, Features, Value, DownloadManager, SplitGenerator, Split

DATA_PATH = r"C:\Users\Drivi\Documents\技术书籍\datasets\瑞金MMC知识图谱构建\0521_new_format"
_DESCRIPTION = "阿里云天池大赛MMC知识图谱构建数据集"
_URLS = {
    "all": DATA_PATH
}

class MmcDataset(GeneratorBasedBuilder):
    VERSION = Version('1.0.0')
    BUILDER_CONFIGS = [BuilderConfig(name="MMC-dataset")]
    DEFAULT_CONFIG_NAME = "MMC-dataset"

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
            description=_DESCRIPTION,
            features=Features(features)
        )
        return info

    def _split_generators(self, dl_manager: DownloadManager):
        data_dir = dl_manager.download_and_extract(_URLS['all'])
        splits = [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={'split': 'train', 'filepath': data_dir}
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={'split': 'test', 'filepath': data_dir}
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={'split': 'validation', 'filepath': data_dir}
            ),
        ]
        return splits

    def _generate_examples(self, **kwargs):
        pass


