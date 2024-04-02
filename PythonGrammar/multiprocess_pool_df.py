"""
本脚本展示了使用进程池时，数据（特别是pandas.DataFrame）的序列化是否传递进入子进程的一些总结。
要点：
1. if __name__ == '__main__' 这个很重要，测试数据的生成在这个代码 之前 还是 之后，有很大的影响；
2. 子进程的生成方式，fork 或者 spawn 也有重要的影响
"""
import os
import psutil
import numpy as np
import pandas as pd
from time import sleep
import random
import multiprocessing as mp
from multiprocessing import Process, Pool, Semaphore, Condition, current_process
from multiprocessing import Queue as MQueue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# psutil.Process(os.getpid()).memory_info()
# print('当前进程的内存使用：{:.2f} MB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

print("pid[{}] - beginning - used memory size: {:.2f} MB".format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
# ---------numpy 随机数----------------
rng = np.random.RandomState(0)
# 每一行 1024/8 个 float64 数，大小刚好 1KB
# a = rng.rand(1024, int(1024/8))
# a.__sizeof__()/(1024*1024)

def generate_df_list():
    """
    生成测试数据.
    这里使用了两种方式生成DF的列表，经过验证，这两种方式对于序列化没有区别
    """
    # 第一种，先产生一个总DF，然后通过索引获取各个 df_part 的数据
    array = rng.rand(1024*(128+256+384+512), int(1024/8))
    df = pd.DataFrame(array)
    df_list = [
        df[0:(1024 * 128)],  # 128MB
        df[(1024 * 128):(1024 * (128 + 256))],  # 256MB
        df[(1024 * (128 + 256)):(1024 * (128 + 256 + 384))],  # 384MB
        df[(1024 * (128 + 256 + 384)):]  # 512MB
    ]
    # 做一次copy的方式
    # df_list = [
    #     df[0:(1024 * 128)].copy(),  # 128MB
    #     df[(1024 * 128):(1024 * (128 + 256))].copy(),  # 256MB
    #     df[(1024 * (128 + 256)):(1024 * (128 + 256 + 384))].copy(),  # 384MB
    #     df[(1024 * (128 + 256 + 384)):].copy()  # 512MB
    # ]
    # print("pid[{}]- copy -  used memory size: {:.2f} MB".format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

    # 第二种，直接分开生成各个 df_part 的数据
    # df_list = [
    #     pd.DataFrame(rng.rand(1024*128, int(1024/8))),  # 128MB
    #     pd.DataFrame(rng.rand(1024*256, int(1024/8))),  # 256MB
    #     pd.DataFrame(rng.rand(1024*384, int(1024/8))),  # 384MB
    #     pd.DataFrame(rng.rand(1024*512, int(1024/8))),  # 512MB
    # ]
    return df_list


def show_df(df_part):
    """
    展示传入的df_part的大小，子进程占用内存大小
    """
    shape = df_part.shape
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print("pid[{}] processing df_part with rows '{}' and using memory: {:.2f} MB".format(os.getpid(), shape[0]/1024, memory))
    # random.random()随机生成0~1之间的浮点数
    sleep_time = random.random()*2
    sleep(sleep_time)
    # print("msg '{}' sleep for {:.4f} second.".format(msg, sleep_time))
    # return sleep_time
    print("="*16)
    return shape


def show_self_df(value):
    """
    随便传入一个 value，子进程中固定生成一个 256MB 的df，展示子进程前后占用的内存大小
    """
    memory_begin = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print("pid[{}] - value '{}' -  begin -  using memory: {:.2f} MB".format(os.getpid(), value, memory_begin))
    a = rng.rand(1024 * 256, int(1024 / 8))
    df_part = pd.DataFrame(a)
    shape = df_part.shape
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print("pid[{}] - value '{}' - end - using memory: {:.2f} MB".format(os.getpid(), value, memory))
    # random.random()随机生成0~1之间的浮点数
    sleep_time = random.random()*2
    sleep(sleep_time)
    # print("msg '{}' sleep for {:.4f} second.".format(msg, sleep_time))
    # return sleep_time
    print("="*16)
    return shape

"""
如果在 if __name__ == '__main__' **之前** 产生测试数据，根据平台会有如下情况：
（1）Windows下，默认使用 spawn 方式生成子进程，此时 if __name__ == '__main__' 之前的代码，
  不仅在主进程中会执行一次，在子进程中也会执行一次，也就是执行了 generate_df_list() —— 这就导致子进程内存中会生成一份多余的 df_list
（2）Linux下，有 fork 和 spawn 两种方法（这里没有研究 forkserver 的方式）：
  2.1 默认的fork模式下，if __name__ == '__main__' 之前的代码在子进程中不会执行，只会从 fork 开始的地方进行父子进程的分叉，
    但是此时会将父进程中的 df_list 继承下来，所以仍然有一份重复数据；
  2.2 spawn模式下，if __name__ == '__main__' 之前的代码 generate_df_list() 也会执行一次（和Windows一样），
    因此会重新生成一份 df_list，导致也有重复数据。 
  总结：Linux下不论是 fork 还是 spawn，每个子进程里都会有一份 df_list 的数据
"""
# ========================================================================
# df_list = generate_df_list()
# for df_part in df_list:
#     print("pid[{}] - generate df_part with rows '{}' memory size: {:.2f} MB".format(os.getpid(), df_part.shape[0]/1024, df_part.__sizeof__() / (1024 * 1024)))
# print("pid[{}]- ready - used memory size: {:.2f} MB".format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
# print("-"*16)
# ========================================================================

"""
下面这段在 if __name__ == '__main__' 之前开启进程池的代码，只能在 fork 模式下执行；
spawn模式下，不论是Windows还是Linux，都会报错。
"""
# mp.set_start_method('spawn')
# print("*" * 8 + " main code: pid[{}] run pool.map ".format(os.getpid()) + "*" * 8 + "\n")
# pool = Pool(2)
# res = pool.map(show_df, df_list)
# print(res)
# # 再开一次进程池
# print("\n" + "+" * 8 + " main code: pid[{}] run pool.map again ".format(os.getpid()) + "+" * 8 + "\n")
# pool = Pool(2)
# res = pool.map(show_df, df_list)
# print(res)

print("before main code....")

if __name__ == '__main__':
    # process_mode = 'spawn'
    # process_mode = 'fork'  # Windows下没有这个模式
    # process_mode = 'forkserver'
    # mp.set_start_method(process_mode)

    print("start main code ....")
    # print("start main code with mode {} ....".format(process_mode))

    """
    如果在 if __name__ == '__main__' **之后** 产生测试数据，根据平台会有如下情况：
    （1）Windows下，默认使用 spawn 方式生成子进程，此时 if __name__ == '__main__' 之后的代码，
      只会在主进程中会执行一次，在子进程中不会执行 —— 此时子进程中就不会有多余的 df_list 数据。
    （2）Linux下，有 fork 和 spawn 两种方法（这里没有研究 forkserver 的方式）：
      2.1 默认的fork模式下，if __name__ == '__main__' 之前的代码在子进程中不会执行，只会从 fork 开始的地方进行父子进程的分叉，
        但是此时会将父进程中的 df_list 继承下来，所以仍然有一份重复数据 —— 和之前一样
      2.2 spawn模式下，if __name__ == '__main__' 之前的代码 也会执行一次（和Windows一样），但不会再次执行  generate_df_list() 了，
        因此没有重复数据了。 
      总结：Linux下， fork 的子进程还是会有重复数据，但是 spawn 的子进程不会有重复数据。
    """
    # ========================================================================
    df_list = generate_df_list()
    for df_part in df_list:
        print("pid[{}] - generate df_part with rows '{}' memory size: {:.2f} MB".format(os.getpid(), df_part.shape[0] / 1024, df_part.__sizeof__() / (1024 * 1024)))
    print("pid[{}]- ready - used memory size: {:.2f} MB".format(os.getpid(), psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    print("-" * 16)
    # ========================================================================

    # 主要使用下面的方式进行研究
    print("*"*8 + " main code: pid[{}] run pool.map ".format(os.getpid()) + "*"*8 + "\n")
    # 查看 Pool 类的实现，可以发现，下面这句执行完之后，子进程就已经启动了，并且使用了一个 queue.SimpleQueue() 来存放后续 map 放入的任务
    pool = Pool(2)
    res = pool.map(show_df, df_list)
    print(res)

    # 研究下再开一次进程池的影响，可以发现，不论是 spawn 还是 fork 模式，上面 __name__ 之后的进程池代码不会被二次执行
    print("\n" + "+"*8 + " main code: pid[{}] run pool.map again ".format(os.getpid()) + "+"*8 + "\n")
    pool = Pool(2)
    res = pool.map(show_df, df_list)
    print(res)

    # 使用 show_self_df 函数排查问题
    # print("*"*8 + " main code: pid[{}] run pool.map ".format(os.getpid()) + "*"*8 + "\n")
    # pool = Pool(2)
    # res = pool.map(show_self_df, [1, 2, 3, 4])
    # print(res)
