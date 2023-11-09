"""
练习visdom的使用
"""
import os
import numpy as np
import pandas as pd
from visdom import Visdom

# 执行下面的练习之前，需要使用 python -m visdom.server 启动visdom的服务.
vis_conf = {
    'server': 'http://localhost',  # 默认参数
    'port': 8097,                  # 默认参数
    # 用户认证
    # 'username': 'myself',
    # 'password': 'visdom'
}
vis = Visdom(**vis_conf)

# 检查是否连接成功
print(vis.check_connection())

# 显示文字
win = vis.text('Hello, Visdom!')
print(type(win))  # 类型是 str，值应该是某个 window 的句柄
# 追加到上面的窗口
vis.text("Hello World !", win=win, append=True)
# opts 是字典形式的参数，可接受的参数参考plotly
vis.text("Hello New World !", opts=dict(title='Hello Text', caption='Caption Text', width=864, height=480))

# 显示图片
vis.image(
    np.random.rand(3, 512, 256),
    opts=dict(title='Random!', caption='How random.')
)

# 散点图
Y = np.random.rand(100)
old_scatter = vis.scatter(
    X=np.random.rand(100, 2),
    Y=(Y[Y > 0] + 1.5).astype(int),
    opts=dict(
        title='Scatter Plot',
        legend=['Didnt', 'Update'],
        xtickmin=-50,
        xtickmax=50,
        xtickstep=0.5,
        ytickmin=-50,
        ytickmax=50,
        ytickstep=0.5,
        markersymbol='cross-thin-open',
    ),
)
# 更新之前的窗口参数
vis.update_window_opts(
    win=old_scatter,
    opts=dict(
        legend=['Apples', 'Pears'],
        xtickmin=0,
        xtickmax=1,
        xtickstep=0.5,
        ytickmin=0,
        ytickmax=1,
        ytickstep=0.5,
        markersymbol='cross-thin-open',
    ),
)

vis.line(
    X=np.arange(20),
    Y=np.random.random(20),
    opts=dict(
        title='Line Plot'
    )
)

# 条形图
vis.bar(
    X=np.abs(np.random.rand(5, 3)),
    opts=dict(
        stacked=True,
        legend=['Facebook', 'Google', 'Twitter'],
        rownames=['2012', '2013', '2014', '2015', '2016']
    )
)

# 关闭之后，会清空当前会话的所有图形
vis.close()