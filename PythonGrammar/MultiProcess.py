"""
Python 并行编程
"""
# 多线程相关
import threading
from queue import Queue
# 多进程相关
import multiprocessing
from multiprocessing import Pool, Semaphore, Condition, Manager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from time import sleep
import random
import os


# --- 基本的线程使用 ---------------------
# 方法一，传入函数
def my_fun(num):
    print("thread " + num + " starting")
    sleep(0.2)
    print("thread " + num + " ending")

# 方法二、继承线程类，并重载run方法
class ThreadFunction(threading.Thread):
    def __init__(self, num):
        super().__init__()
        self.num = num

    # def run(self) -> None:
    #     super().run()
    def run(self):
        print("thread " + self.num + " starting")
        sleep(0.2)
        print("thread " + self.num + " ending")

# ---- 基本的进程使用 --------------------
# 方法一，传入函数, 函数同线程的 my_fun
# 方法二、继承进程类，并重载run方法
class ProcessFunction(multiprocessing.Process):
    def __init__(self, num):
        super().__init__()
        self.num = num

    # def run(self) -> None:
    #     super().run()
    def run(self):
        print("process " + self.num + " starting")
        sleep(0.2)
        print("process " + self.num + " ending")


# ----------------进程或线程池的使用---------------------------------------
def worker(msg, level):
    print("{} {} starting, {} num is: {}.".format(level, msg, level, os.getpid()))
    # random.random()随机生成0~1之间的浮点数
    sleep_time = random.random()*5
    time.sleep(sleep_time)
    print("{} {} sleep for {:.4f} second.".format(level, msg, sleep_time))
    return sleep_time


# ---------------- 带有锁 的线程同步 ---------------------
# 线程里迭代的次数
COUNT = 2000
shared_resource_with_lock = 0
shared_resource_without_lock = 0
# 线程锁
# thread_lock = threading.Lock()
thread_lock = threading.RLock()

# 带有锁管理的 两个线程函数
def increment_with_lock():
    # 引入全局变量，这一句必须要有，它表示要 读并写 函数外部的变量
    global shared_resource_with_lock
    # COUNT 这个全局变量只是读，所以不需要 global 关键字
    for i in range(COUNT):
        thread_lock.acquire()
        print("increment with lock: ", shared_resource_with_lock)
        shared_resource_with_lock += 1
        thread_lock.release()
        sleep(0.5)

def decrement_with_lock():
    global shared_resource_with_lock
    for i in range(COUNT):
        thread_lock.acquire()
        print("decrement with lock: ", shared_resource_with_lock)
        shared_resource_with_lock -= 1
        thread_lock.release()
        sleep(0.5)

# 没有锁管理 的两个线程函数
def increment_without_lock():
    global shared_resource_without_lock
    for i in range(COUNT):
        print("increment without lock: ", shared_resource_without_lock)
        shared_resource_without_lock += 1
        sleep(0.1)

def decrement_without_lock():
    global shared_resource_without_lock
    for i in range(COUNT):
        print("decrement without lock: ", shared_resource_without_lock)
        shared_resource_without_lock -= 1
        sleep(0.1)

# ---------------------------------------------------------------
# ------------ 生产者-消费者 模型的 多种实现 ------------------------
# ---------------------------------------------------------------

# ----------多线程+队列 的生产者-消费者模型---------------------
class Producer(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        for i in range(10):
            item = random.randint(0, 256)
            self.queue.put(item)
            print('Producer notify: item {} append to queue by {}'.format(item, self.name))
            sleep(0.5)


class Consumer(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            item = self.queue.get()
            print("Consumer notify: item {} popped from queue by {}".format(item, self.name))
            self.queue.task_done()


# -----------多进程+队列 的生产者-消费者模型--------------------
# 这里没有使用子类继承的方式
def producer(name, queue):
    print("producer " + name + " is running")
    i = 0
    while i < 20:
        item = random.randint(0, 50)
        queue.put(item)
        print("producer {} putting item {} successfully".format(name, item))
        i = i+1
        sleep(0.5)


def consumer(name, queue):
    while True:
        # print("consumer {} get Queue size : {}".format(name, queue.qsize()))
        item = queue.get()
        # item = queue.get(block=False, timeout=1000)
        if item:
            print("consumer {} getting item {}.".format(name, item))
            sleep(0.5)


# ------ 线程同步：信号量 实现的 生产者-消费者 模型 -----------------
# 信号量的初始值=0，而不是1
semaphore = threading.Semaphore(0)

def producer_sem(name):
    # item是一个全局变量
    global item
    for i in range(0, 10):
        sleep(0.5)
        item = random.randint(0, 256)
        print("producer {} notify: produced item {}".format(name, item))
        # 释放信号量，对其内部的计数器 +1
        semaphore.release()

def consumer_sem(name):
    while True:
        sleep(1)
        print("consumer {} is waiting.".format(name))
        # 获取信号量
        semaphore.acquire()
        print("consumer {} notify: consumed item {}".format(name, item))


# ------ 线程同步：条件变量 实现的 生产者-消费者模型 ------------------------------
items = []
condition = threading.Condition()

class Consumer_cond(threading.Thread):
    def __init__(self):
        super().__init__()

    def consume(self):
        global condition
        global items
        condition.acquire() # 这个锁必须要在下面的判断条件之前，保证条件和 wait 操作的原子性
        if len(items) == 0:
            condition.wait()
            print("Consumer notify: no item to consume")
        item = items.pop()
        # 执行 notify 时必须要持有锁，所以要在这之后释放锁
        condition.notify()
        condition.release() # 释放锁
        # print操作不用放在锁里面
        print("Consumer notify: consume 1 item '{}'".format(item))
        print("Consumer notify: items to be comsumed are {}".format(len(items)))

    def run(self):
        for i in range(20):
            sleep(0.5)
            self.consume()


class Producer_cond(threading.Thread):
    def __init__(self):
        super().__init__()

    def produce(self):
        global condition
        global items
        condition.acquire() # 条件判断前必须要加锁
        if len(items) == 10:
            condition.wait()
            print("Producer notify: items produced are {}".format(len(items)))
            print("Producer notify: stop the production !")
        items.append(random.randint(0, 256))
        condition.notify() # notify 操作时必须要持有锁，所以 release 要在此之后
        condition.release()
        print("Producer notify: total items are {}".format(len(items)))

    def run(self):
        for i in range(0, 20):
            sleep(0.5)
            self.produce()


if __name__ == "__main__":
    # ---------基本线程使用---------------
    # # 创建线程
    # t1 = threading.Thread(target=my_fun, args=("thread-1",), name="thread-1")
    # t2 = ThreadFunction(num="thread-2")
    # # 开始线程
    # t1.start(), t2.start()
    # # join表示主进程在此处阻塞，等待线程执行结束后再继续
    # t1.join(), t2.join()

    # ------- 进程的基本使用 ------------------
    # t1 = multiprocessing.Process(target=my_fun, args=("process-1",), name="process-1")
    # t2 = ThreadFunction(num="process-2")
    # t1.start(), t2.start()
    # t1.join(), t2.join()

    # --------进程池的使用------------------
    # future_list = []
    # with ThreadPoolExecutor(max_workers=5) as executor:
    #     for i in range(10):
    #         future = executor.submit(worker, i+1, "Thread")
    #         future_list.append(future)
    # for future in future_list:
    #     print("future.result: ", future.result())

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
    # pool = Pool(4)
    # pool.map(worker, range(0, 12))
    # print("----start----")
    # pool.close()  # 关闭进程池，关闭后po不再接收新的请求
    # pool.join()  # 等待po中所有子进程执行完成，再执行下面的代码,可以设置超时时间join(timeout=)
    # print("-----end-----")


    # --------------- 线程锁的使用 -----------------------
    # t1 = threading.Thread(target=increment_with_lock)
    # t2 = threading.Thread(target=decrement_with_lock)
    # t3 = threading.Thread(target=increment_without_lock)
    # t4 = threading.Thread(target=decrement_without_lock)
    # t1.start(), t2.start(), t1.join(), t2.join()
    # print("------------shared_resource_with_lock: ", shared_resource_with_lock, "-----------")
    # t3.start(), t4.start(), t3.join(), t4.join()
    # print("------------shared_resource_with_no_lock: ", shared_resource_without_lock, "-----------")

    # ---------------------------------------------------------------
    # ------------- 生产者-消费者 模型的 多种实现 ------------------------
    # ---------------------------------------------------------------

    # --------多线程+队列 的 生产者-消费者 模型 --------------
    # queue = Queue()
    # t1 = Producer(queue)
    # t2 = Consumer(queue)
    # t3 = Consumer(queue)
    # t1.start(), t2.start(), t3.start()
    # t1.join(), t2.join(), t3.join()

    # --------多进程+队列 的 生产者-消费者 模型--------------
    # queue = multiprocessing.Queue()
    # p = multiprocessing.Process(target=producer, args=('Producer-1', queue))
    # c1 = multiprocessing.Process(target=consumer, args=('Consumer-1', queue))
    # c2 = multiprocessing.Process(target=consumer, args=('Consumer-2', queue))
    # p.start(), c1.start(), c2.start()
    # p.join(), c1.join(), c2.join()

    # ----- 线程同步：信号量 实现的 消费者-生产者 模型 ----------------
    # p = threading.Thread(target=producer_sem, args=('p-1', ))
    # c1 = threading.Thread(target=consumer_sem, args=('c-1', ))
    # c2 = threading.Thread(target=consumer_sem, args=('c-2', ))
    # p.start(), c1.start(), c2.start()
    # p.join(), c1.join(), c2.join()
    # print("--------------finished--------------")

    # ----- 线程同步：条件变量 实现的 消费者-生产者 模型 -----------------
    p = Producer_cond()
    c = Consumer_cond()
    p.start(), c.start()
    p.join(), c.join()

