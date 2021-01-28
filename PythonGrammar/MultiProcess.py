"""
Python 并行编程
"""
import threading
import multiprocessing
from multiprocessing import Pool
import time
from time import sleep
import random
import os


def my_fun(name):
    print("thread " + name + " starting")
    sleep(1)
    print("thread " + name + " ending")


# 创建线程
# t1 = threading.Thread(target=my_fun, args=("thread-1",), name="thread-1")
# t2 = threading.Thread(target=my_fun, args=("thread-2",), name="thread-2")
#

# 开始线程
# t1.start()
# t2.start()

# join表示主进程在此处阻塞，等待线程执行结束后再继续
# t1.join()
# t2.join()

# -----------------------------------------------------------
# 多线程的生产者-消费者模型






# -----------------------------------------------------------
# 多进程的生产者-消费者模型

def producer(name, q):
    print("producer " + name + " is running")
    i = 0
    while i <= 10:
        # item = random.randint(0, 50)
        item = i
        q.put(item)
        print("producer putting item {} successfully".format(item))
        i = i+1
        sleep(1)


def consumer(name, q):
    while True:
        print("consumer {} get Queue size : {}".format(name, q.qsize()))
        item = q.get()
        if item:
            print("consumer {} getting item {}.".format(name, item))
        else:
            print("consumer {} getting -----Nothing------.".format(name))
        sleep(1)


# -----------------------------------------------
# 进程池的使用
def worker(msg):
    print("进程 {} 开始执行,进程号为 {}.".format(msg, os.getpid()))
    # random.random()随机生成0~1之间的浮点数
    sleep_time = random.random()*5
    time.sleep(sleep_time)
    print("进程 ", msg, " 睡眠时间为：{:.4f}.".format(sleep_time))



if __name__ == '__main__':
    # 多进程+队列 的 生产者-消费者 模型
    # q = multiprocessing.Queue()
    # p = multiprocessing.Process(target=producer, args=('Random Number', q))
    # c1 = multiprocessing.Process(target=consumer, args=('Consumer-1', q))
    # c2 = multiprocessing.Process(target=consumer, args=('Consumer-2', q))
    # p.start()
    # c1.start()
    # c2.start()
    # p.join()
    # c1.join()
    # c2.join()

    # 进程池的使用

    # 一次提交一个进程
    # po = Pool(3)  # 定义一个进程池，最大进程数3
    # for i in range(0, 10):
    #     po.apply_async(worker, (i,))
    #
    # print("----start----")
    # po.close()  # 关闭进程池，关闭后po不再接收新的请求
    # po.join()  # 等待po中所有子进程执行完成，再执行下面的代码,可以设置超时时间join(timeout=)
    # print("-----end-----")

    # 一次提交多个进程，使用进程池中的所有进程并行执行某个函数
    pool = Pool(4)
    pool.map(worker, range(0, 12))
    print("----start----")
    pool.close()  # 关闭进程池，关闭后po不再接收新的请求
    pool.join()  # 等待po中所有子进程执行完成，再执行下面的代码,可以设置超时时间join(timeout=)
    print("-----end-----")











