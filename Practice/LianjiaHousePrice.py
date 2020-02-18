import numpy as np
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# 链家房价爬虫
class LianjiaCrawer():

    def __init__(self):
        self.base_url = r'https://www.lianjia.com/city/'
        self.driver = webdriver.Chrome()
        self.pageNum = 10

    def main(self):
        # 得到要爬取的城市以及房屋类型的首页信息
        type = self.getCityHomePage()
        data_list = []
        page = 1
        if type == '1':
            while page <= self.pageNum:
                print('[Info] processing page {};'.format(page))
                df_list = self.parse_page_xinfang()
                data_list.extend(df_list)
                self.turnPage()
                page += 1
        if type == '2':
            while page <= self.pageNum:
                print('[Info] processing page {};'.format(page))
                df_list = self.parse_page_ershoufang()
                data_list.extend(df_list)
                self.turnPage()
                page += 1
        data_df = pd.DataFrame(data_list)
        data_df.head()


    def getCityHomePage(self):
        # base_url = r'https://www.lianjia.com/city/'
        # driver = webdriver.Chrome()
        self.driver.get(self.base_url)
        search_city = self.driver.find_element_by_css_selector('.search_wrapper input')
        city = input("请输入城市名称：")
        search_city.send_keys(city)
        # search_city.clear()
        search_city.send_keys(Keys.ENTER)
        # 上面这句执行完之后会新开一个窗口
        # driver.window_handles
        self.driver.switch_to.window(self.driver.window_handles[1])
        # driver.current_url
        # driver.current_window_handle
        search_box_wrap = self.driver.find_element_by_css_selector('.search-box-wrap')
        type = input("请问查找的是新房还是二手房（1=新房，2=二手房）：")
        if type == '1':
            # 找到新房的菜单
            tab = search_box_wrap.find_element_by_css_selector('[actdata="channel=xinfang"')
            tab.click()
            print('进入新房菜单')
            search_box_wrap.find_element_by_id('findHouse').click()
        if type == '2':
            # 找到二手房的菜单
            tab = search_box_wrap.find_element_by_css_selector('[actdata="channel=ershoufang"')
            tab.click()
            print('进入二手房菜单')
            search_box_wrap.find_element_by_id('findHouse').click()
        return type

    def turnPage(self):
        pageBox = self.driver.find_element_by_class_name('page-box.fr')
        pageData_str = pageBox.find_element_by_css_selector('.page-box.house-lst-page-box').get_attribute('page-data')
        pageData_dict = eval(pageData_str)
        nextPage = pageBox.find_elements_by_css_selector('a')[-1]
        if pageData_dict['curPage'] < 100:
            nextPage.click()

    def parse_page_xinfang(self):
        resblock_list = self.driver.find_elements_by_css_selector('.resblock-list.post_ulog_exposure_scroll.has-results')
        block_parse_list = [self.parse_block_xinfang(block) for block in resblock_list]
        # block_parse_list = [parse_block_xinfang(block) for block in resblock_list]
        return block_parse_list

    def parse_block_xinfang(block):
        # block是.resblock-list.post_ulog_exposure_scroll.has-results选择的元素
        block_info = {'name': block.find_element_by_css_selector('.resblock-name .name').text,
                      'type': block.find_element_by_css_selector('.resblock-name .resblock-type').text,
                      'sale_status': block.find_element_by_css_selector('.resblock-name .sale-status').text,
                      'location': block.find_element_by_css_selector('.resblock-location').text,
                      'room': block.find_element_by_css_selector('.resblock-room').text,
                      'area': block.find_element_by_css_selector('.resblock-area').text,
                      'tag': block.find_element_by_css_selector('.resblock-tag').text,
                      'price': block.find_element_by_css_selector('.resblock-price .main-price').text,
                      'total': block.find_element_by_css_selector('.resblock-price .second').text}
        return block_info

    def parse_page_ershoufang(self):
        resblock_list = self.driver.find_elements_by_css_selector('.clear.LOGCLICKDATA')
        block_parse_list = [self.parse_block_ershoufang(block) for block in resblock_list]
        # block_parse_list = [parse_block_ershoufang(block) for block in resblock_list]
        return block_parse_list

    def parse_block_ershoufang(block):
        # block是.clear.LOGCLICKDATA选择的元素
        block_info = {'title': block.find_element_by_css_selector('.info.clear .title a').text,
                      'position': block.find_element_by_css_selector('.info.clear .positionInfo').text,
                      'houseInfo': block.find_element_by_css_selector('.info.clear .houseInfo').text,
                      'tag': block.find_element_by_css_selector('.info.clear .tag').text,
                      'price': block.find_element_by_css_selector('.info.clear .unitPrice').text,
                      'total': block.find_element_by_css_selector('.info.clear .totalPrice').text}
        return block_info


crawer = LianjiaCrawer()
crawer.main()



# driver = webdriver.Chrome()
# url = r'https://hf.fang.lianjia.com/loupan/'
# driver.get(url)
# eles = driver.find_elements_by_css_selector('.resblock-list.post_ulog_exposure_scroll.has-results')
# eles[0]
# eles.__len__()
# ele = eles[0]
# ele.find_element_by_css_selector('.resblock-name .name').text
# ele.find_element_by_css_selector('.resblock-name .resblock-type').text
# ele.find_element_by_css_selector('.resblock-location').text
# ele.find_element_by_css_selector('.resblock-room').text
#
# def parse(ele):
#     info = {}
#     info['name'] = ele.find_element_by_css_selector('.resblock-name .name').text
#     info['type'] = ele.find_element_by_css_selector('.resblock-name .resblock-type').text
#     info['sale_status'] = ele.find_element_by_css_selector('.resblock-name .sale-status').text
#     info['location'] = ele.find_element_by_css_selector('.resblock-location').text
#     info['room'] = ele.find_element_by_css_selector('.resblock-room').text
#     info['area'] = ele.find_element_by_css_selector('.resblock-area').text
#     info['tag'] = ele.find_element_by_css_selector('.resblock-tag').text
#     info['price'] = ele.find_element_by_css_selector('.resblock-price .main-price').text
#     info['total'] = ele.find_element_by_css_selector('.resblock-price .second').text
#     return info
#
# for ele in eles:
#     info = parse(ele)
#     print(info)
#
# info_list = [parse(ele) for ele in eles]
# info_df = pd.DataFrame(info_list)
