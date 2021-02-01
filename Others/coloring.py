# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:21:04 2019

@author: cm
"""

import numpy as np,pandas as pd,matplotlib.pyplot as plt
import os

# 读入涂色数据
filedir = "/Users/danielzhang/Documents/data/coloring_data"
filelist = os.listdir(filedir)
filepath = [filedir + '/'+ path  for path in filelist]
colnames = ["country","ver","pkg_name","sys_type","pic_id","block_num","is_vip_img","pc_texture","img_name","filetag","img_finish_time","img_status","buy_quantity","is_newuser","is_vip_user","location_app","action","user_id"]
coltypes = ["str"]*5 + ["float64"] + ["str"]*12
coltypesdict = dict(zip(colnames,coltypes))
coloring_origin = pd.read_csv(filepath[0],header=None,names = colnames,dtype = coltypesdict)
for i in range(1,len(filepath)):
    df = pd.read_csv(filepath[i],header=None,names = colnames,dtype = coltypesdict)
    coloring_origin = pd.concat([coloring_origin,df])
    
# 读取tag数据
tagfilepath = "/Users/danielzhang/Documents/data/coloring_tag.csv"
tag = pd.read_csv(tagfilepath,header=0,usecols = [1,2])
tag = tag.astype("str")

# 只考虑美国区的iOS用户
coloring = coloring_origin[(coloring_origin["country"]=="US") & (coloring_origin["sys_type"] == "ios")]


#区分新包和VIE
#VIE新包包名：color.by.number.coloring.games  
#老包iOS包名：coloring.art.color.by.number.app
coloring_old = coloring[coloring.pkg_name == "coloring.art.color.by.number.app"]
coloring_vie = coloring[coloring.pkg_name == "color.by.number.coloring.games"]

#old_vs_class = coloring_old.groupby(['filetag','action'],as_index=False).size().reset_index(name = 'count')


# 统计 展示->点击 的漏斗图--------------------------------
# 只有老包里面有展示的记录，所以这一步只需要分析老包
coloring_old['action'].unique()
coloring_vie['action'].unique()

# 统计不同地点的展示、点击数
coloring_old['location_app'].unique()
#
show2click_1 = coloring_old[coloring_old['action'] == '1']
show2click_1.location_app.replace(to_replace={'4':'3'},inplace = True)
show2click_1['location_app'].unique()


