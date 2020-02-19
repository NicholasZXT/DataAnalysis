import numpy as np
import pandas as pd
import requests
import json
import jsonpath
import bs4


# 没有这个headers会被豆瓣判定为机器人，返回的状态码是418
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36'
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

#TODO ----------拉勾网ajax方式-----------
lagou_url = 'https://www.lagou.com/jobs/positionAjax.json'
# 使用sublime将header批量处理成字典的形式
# 正则查找：^(.*):\s(.*)$
# 分组替换："\1": "\2",
# headers
headers = {"Accept": "application/json, text/javascript, */*; q=0.01",
           "Accept-Encoding": "gzip, deflate, br",
           "Accept-Language": "zh,zh-CN;q=0.9,en;q=0.8,zh-TW;q=0.7",
           "Connection": "keep-alive",
           "Content-Length": "64",
           "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
           "Cookie": "JSESSIONID=ABAAAECAAFDAAEHCB48339E450E51DBF51FEFB46F5A443B; WEBTJ-ID=20200211220819-17034944d940-0b125a24dabf86-1d346652-1296000-17034944d957f4; _ga=GA1.2.1722286090.1581430100; _gid=GA1.2.267221363.1581430100; user_trace_token=20200211220832-54bf5ab8-470b-4f8d-836b-61b62c3e7f42; LGUID=20200211220832-d5622fbd-2488-42ff-a2e9-79a9231eefbb; Hm_lvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1581430100,1581513720; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%22170399148c2e6-0bfdab5908f68f-1d346652-1296000-170399148c3687%22%2C%22%24device_id%22%3A%22170399148c2e6-0bfdab5908f68f-1d346652-1296000-170399148c3687%22%7D; LGSID=20200213081314-b489331d-67bb-475c-a91b-0550deb409a7; PRE_UTM=; PRE_HOST=; PRE_SITE=; PRE_LAND=https%3A%2F%2Fwww.lagou.com%2Fjobs%2Flist%5F%25E6%2595%25B0%25E6%258D%25AE%25E5%2588%2586%25E6%259E%2590%25E5%25B8%2588%2Fp-city%5F2%3Fpx%3Ddefault%23filterBox; _gat=1; index_location_city=%E5%8C%97%E4%BA%AC; X_HTTP_TOKEN=549323d74701acf47803551851af51e032eb9b2635; LGRID=20200213081807-bd2cb318-acd6-42c8-8dba-14acc3fc1465; Hm_lpvt_4233e74dff0ae5bd0a3d81c6ccf756e6=1581553074; SEARCH_ID=7eff3a014cdc45caa0b52e9beaa670c6",
           "DNT": "1",
           "Host": "www.lagou.com",
           "Origin": "https://www.lagou.com",
           "Referer": "https://www.lagou.com/jobs/list_%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E5%B8%88/p-city_2?px=default&district=%E6%9C%9D%E9%98%B3%E5%8C%BA",
           "Sec-Fetch-Dest": "empty",
           "Sec-Fetch-Mode": "cors",
           "Sec-Fetch-Site": "same-origin",
           "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36",
           "X-Anit-Forge-Code": "0",
           "X-Anit-Forge-Token": "None",
           "X-Requested-With": "XMLHttpRequest"}
# 查询参数
params = {
    "px": "default",
    "city": "北京",
    "needAddtionalResult": "false"}
# 表单数据
form_data = {
    "first": "true",
    "pn": "1",
    "kd": "数据分析师"}
# 注意这里使用的是post方法而不是get方法
r = requests.post(lagou_url, params=params, data=form_data, headers=headers)
r.status_code
# 这个contents数据拿到的是表单数据，json格式，不是html
r_json = r.content.decode('utf-8')
r_dict = json.loads(r_json)
result_json_list = jsonpath.jsonpath(r_dict, "$.content.positionResult.result")[0]
df = pd.DataFrame(result_json_list)

#TODO ------ 练习使用Selenium爬取拉勾网 ---------
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
# 开启浏览器
browser = webdriver.Chrome(executable_path='/usr/local/chromedriver/chromedriver')
# 打开网页
# browser.get("https://www.baidu.com")
browser.get("https://www.lagou.com")
# 查看当前的网页地址
# browser.current_url
# browser.refresh()
# 这里对于的class名称显示的是 tab focus
ele = browser.find_element_by_class_name('tab.focus')
ele.text
ele.click()
input = browser.find_element_by_id('search_input')
# input.tag_name
input.send_keys("python")
input.send_keys(Keys.ENTER)
# 上面两句也可以写成
input.send_keys(r"python\n")
ele = browser.find_element_by_class_name('body-btn')
ele.text
ele.click()
wait = WebDriverWait(browser, 10)

browser.quit()

