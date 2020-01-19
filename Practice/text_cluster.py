

import numpy as np
import pandas as pd
import jieba
# 用于词性标注
import jieba.posseg as psg

sentence = "中文分词是自然语言处理的第一步"

# 分词
word_list = jieba.cut(sentence)
# 词性标注
word_property = psg.cut(sentence)
