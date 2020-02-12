import numpy as np
import pandas as pd
import requests
import json
import jsonpath
import bs4


# 没有这个headers会被豆瓣判定为机器人，返回的状态码是418
headers = {
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36'
}
douban_url = 'https://movie.douban.com/tag/#/?sort=S&range=0,10&tags=%E5%89%A7%E6%83%85'
r = requests.get(douban_url, headers=headers)
r.status_code
r.encoding
html = r.content

# 下面这个是豆瓣电影动态页面加载新的电影信息时请求的数据url
json_url = 'https://movie.douban.com/j/new_search_subjects?sort=S&range=0,10&tags=&start=60&genres=%E5%89%A7%E6%83%85'
r = requests.get(douban_url, headers=headers)
r.status_code
r.encoding
# 获取内容
json_str = r.content
# 将上面这个json字符串转成dict，然后才能被jsonpath处理
json_dict = json.loads(json_str)
# 使用jsonpath提取title信息
title_list = jsonpath.jsonpath(json_dict, "$..title")

