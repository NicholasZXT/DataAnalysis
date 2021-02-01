import numpy as np
import pandas as pd
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException

# 链家房价爬虫
class LianjiaCrawer():

    def __init__(self):
        self.base_url = r'https://www.lianjia.com/city/'
        self.driver = webdriver.Chrome()
        # 下面这个参数是用于新房爬取时的等待，因为二手房的页面加载较快，新房页面加载较慢，不设置这个参数的话
        # 会导致找不到元素
        self.driver.implicitly_wait(5)
        self.pageNum = 10

    def main(self):
        # 得到要爬取的城市以及房屋类型的首页信息
        city = input("请输入城市名称：")
        type = input("请问查找的是新房还是二手房（1=新房，2=二手房）：")
        type = self.getCityHomePage(city, type)
        data_list = []
        page = 1
        #TODO 下面要注意的是，链家的新房和二手房页面布局和css样式类不一样，所以要分别处理
        if type == '1':
            while page <= self.pageNum:
                print('[Info] processing page {};'.format(page))
                df_list = self.parse_page_xinfang()
                data_list.extend(df_list)
                self.turnPage_xinfang()
                page += 1
        if type == '2':
            while page <= self.pageNum:
                print('[Info] processing page {};'.format(page))
                df_list = self.parse_page_ershoufang()
                data_list.extend(df_list)
                self.turnPage_ershoufang()
                page += 1
        data_df = pd.DataFrame(data_list)
        # data_df.to_excel('LianjiaHousePrice.xlsx',index=False)
        print(data_df.shape)
        print(data_df.head())
        self.driver.quit()


    def getCityHomePage(self,city,type):
        # base_url = r'https://www.lianjia.com/city/'
        # driver = webdriver.Chrome()
        self.driver.get(self.base_url)
        search_city = self.driver.find_element_by_css_selector('.search_wrapper input')
        search_city.send_keys(city)
        # search_city.clear()
        #TODO 这里要等待一会，否则输入了城市，服务器没有返回响应，就会导致显示未找到该城市
        time.sleep(2)
        search_city.send_keys(Keys.ENTER)
        time.sleep(2)
        # 上面这句执行完之后会新开一个窗口
        # print(self.driver.window_handles)
        self.driver.switch_to.window(self.driver.window_handles[1])
        # driver.current_url
        # driver.current_window_handle
        search_box_wrap = self.driver.find_element_by_css_selector('.search-box-wrap')
        if type == '1':
            # 找到新房的菜单
            tab = search_box_wrap.find_element_by_css_selector('[actdata="channel=xinfang"')
            tab.click()
            print('进入新房菜单')
            search_box_wrap.find_element_by_css_selector('.text.left.txt.searchVal.autoSuggest').send_keys(Keys.ENTER)
            #TODO 不能采用下面这种寻找搜索按钮然后点击的方式，因为链家页面加载出来时，会出现一个通告，
            # 这个通告会遮住搜索按钮，导致无法执行点击这个动作
            #print(search_box_wrap.find_element_by_id('findHouse').text)
            #search_box_wrap.find_element_by_id('findHouse').click()
        if type == '2':
            # 找到二手房的菜单
            tab = search_box_wrap.find_element_by_css_selector('[actdata="channel=ershoufang"')
            tab.click()
            print('进入二手房菜单')
            search_box_wrap.find_element_by_css_selector('.text.left.txt.searchVal.autoSuggest').send_keys(Keys.ENTER)
            # print(search_box_wrap.find_element_by_id('findHouse').text)
            # search_box_wrap.find_element_by_id('findHouse').click()
        return type

    def turnPage_ershoufang(self):
        pageBox = self.driver.find_element_by_css_selector('.page-box.fr')
        pageData_str = pageBox.\
            find_element_by_css_selector('.page-box.house-lst-page-box').get_attribute('page-data')
        pageData_dict = eval(pageData_str)
        nextPage = pageBox.find_elements_by_css_selector('a')[-1]
        print(nextPage.text)
        if pageData_dict['curPage'] < 100:
            nextPage.click()

    def turnPage_xinfang(self):
        pageBox = self.driver.find_element_by_css_selector('.page-box')
        # pageData_str = pageBox.get_attribute('data-current')
        # pageData_dict = eval(pageData_str)
        nextPage = pageBox.find_elements_by_css_selector('a')[-1]
        print(nextPage.text)
        nextPage.click()
        # if pageData_dict['curPage'] < 100:
        #     nextPage.click()

    def parse_page_xinfang(self):
        resblock_list = self.driver. \
            find_elements_by_css_selector('.resblock-list-wrapper > li')
        # print(len(resblock_list))
        # resblock_list = self.driver.\
        #     find_elements_by_css_selector('.resblock-list.post_ulog_exposure_scroll.has-results')
        block_parse_list = [self.parse_block_xinfang(block) for block in resblock_list]
        # block_parse_list = [parse_block_xinfang(block) for block in resblock_list]
        return block_parse_list

    def parse_block_xinfang(self, block):
        # block是.resblock-list.post_ulog_exposure_scroll.has-results选择的元素
        print(block.find_element_by_css_selector('.resblock-name .name').text)
        block_info = {'name': block.find_element_by_css_selector('.resblock-name .name').text,
                      'type': block.find_element_by_css_selector('.resblock-name .resblock-type').text,
                      'sale_status': block.find_element_by_css_selector('.resblock-name .sale-status').text,
                      'location': block.find_element_by_css_selector('.resblock-location').text,
                      'room': block.find_element_by_css_selector('.resblock-room').text,
                      'area': block.find_element_by_css_selector('.resblock-area').text,
                      'tag': block.find_element_by_css_selector('.resblock-tag').text,
                      'price': block.find_element_by_css_selector('.resblock-price .main-price').text
                      #TODO 新房的页面里，不是每一个房源都有总价这个信息，所以下面这个会报错
                      #'total': block.find_element_by_css_selector('.resblock-price .second').text
                      }
        # 采用try/excerpt来解决上面那个问题
        try :
            total = block.find_element_by_css_selector('.resblock-price .second').text
        except NoSuchElementException:
            total = ""
        block_info['total'] = total
        return block_info

    def parse_page_ershoufang(self):
        resblock_list = self.driver.find_elements_by_css_selector('.clear.LOGCLICKDATA')
        block_parse_list = [self.parse_block_ershoufang(block) for block in resblock_list]
        # block_parse_list = [parse_block_ershoufang(block) for block in resblock_list]
        return block_parse_list

    def parse_block_ershoufang(self,block):
        # block是.clear.LOGCLICKDATA选择的元素
        print(block.find_element_by_css_selector('.info.clear .title a').text)
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
