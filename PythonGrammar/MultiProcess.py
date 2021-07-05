"""
Python 并行编程
"""
import os
from time import sleep
import random
# 多线程相关
import threading
from queue import Queue as TQueue  # 这个队列是线程安全的
# 多进程相关
from multiprocessing import Process, Pool, Semaphore, Condition
from multiprocessing import Queue as MQueue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# =================== 基本的线程使用 ============================
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


# if __name__ == "__main__":
    # ---------基本线程使用---------------
    # # 创建线程
    # t1 = threading.Thread(target=my_fun, args=("thread-1",), name="thread-1")
    # t2 = ThreadFunction(num="thread-2")
    # # 开始线程
    # t1.start(), t2.start()
    # # join表示主进程在此处阻塞，等待线程执行结束后再继续
    # t1.join(), t2.join()


# =================== 基本的进程使用 =============================
# 方法一，传入函数, 函数同线程的 my_fun
# 方法二、继承进程类，并重载run方法
class ProcessFunction(Process):
    def __init__(self, num):
        super().__init__()
        self.num = num

    # def run(self) -> None:
    #     super().run()
    def run(self):
        print("process " + self.num + " starting")
        sleep(0.2)
        print("process " + self.num + " ending")


# if __name__ == "__main__":
    # ------- 进程的基本使用 ------------------
    # t1 = multiprocessing.Process(target=my_fun, args=("process-1",), name="process-1")
    # t2 = ThreadFunction(num="process-2")
    # t1.start(), t2.start()
    # t1.join(), t2.join()


# ============================================================
# ----------------进程或线程池的使用-----------------------------
# ============================================================
def worker(level, msg):
    print("{} of {} starting, process id is: {}.".format(level, msg, os.getpid()))
    # random.random()随机生成0~1之间的浮点数
    sleep_time = random.random()*5
    sleep(sleep_time)
    print("{} of {} sleep for {:.4f} second.".format(level, msg, sleep_time))
    return sleep_time


# if __name__ == "__main__":
    # -------- Pool 的使用------------------

    # 使用 apply 方法，一次提交一个进程，并阻塞直到子进程执行完
    # 这种方式不必使用 join 等待，返回的结果就是直接是函数的返回结果
    # with Pool(3) as pool:
    #     res = pool.apply(worker, ('process', 'worker-1'))
    #     print("worker-1 is done, res is :", res)
    #     res = pool.apply(worker, ('process', 'worker-2'))
    #     print("worker-2 is done, res is :", res)
    #     res = pool.apply(worker, ('process', 'worker-3'))
    #     print("worker-3 is done, res is :", res)

    # 使用 apply_async 进行异步调用，返回的 res 是一个 pool.ApplyResult 对象
    # 必须要 使用 join 方法开启任务
    # pool = Pool(3)
    # res = pool.apply_async(worker, ('process', 'worker-1'))
    # print("worker-1 is done, res is: ", res)
    # res = pool.apply_async(worker, ('process', 'worker-2'))
    # print("worker-2 is done, res is: ", res)
    # res = pool.apply_async(worker, ('process', 'worker-3'))
    # print("worker-3 is done, res is: ", res)
    # print("----start----")
    # pool.close()  # 关闭进程池，关闭后po不再接收新的请求
    # pool.join()  # 等待po中所有子进程执行完成，再执行下面的代码,可以设置超时时间join(timeout=)
    # print("-----end-----")

    # 使用 map 方法一次提交多个进程，使用进程池中的所有进程并行执行某个函数 ----- 不好用
    # pool = Pool(3)
    # pool.map(worker, [('process', 'worker-1')])
    # # pool.map(worker, [('process', 'worker-1'), ('process', 'worker-2'), ('process', 'worker-3')])
    # print("----start----")
    # pool.close()  # 关闭进程池，关闭后po不再接收新的请求
    # pool.join()  # 等待po中所有子进程执行完成，再执行下面的代码,可以设置超时时间join(timeout=)
    # print("-----end-----")

    # --------- 使用  concurrent.futures ---------------------
    # future_list = []
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     for i in range(3):
    #         future = executor.submit(worker, "Thread",  i+1)
    #         future_list.append(future)
    # for future in future_list:
    #     print("future.result: ", future.result())
    # with ProcessPoolExecutor(max_workers=3) as executor:
    #     for i in range(3):
    #         future = executor.submit(worker, "Process",  i+1)
    #         future_list.append(future)
    # for future in future_list:
    #     print("future.result: ", future.result())


# ============================================================
# ---------------- 线程同步 -----------------------------------
# ============================================================
# ---------------- 带有锁 的线程同步 ---------------------
# 线程里迭代的次数
COUNT = 2000
shared_resource_with_lock = 0
shared_resource_without_lock = 0
# 线程锁
# thread_lock = threading.Lock()
thread_lock = threading.RLock()  # 可重入锁

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


# if __name__ == "__main__":
    # --------------- 线程锁的使用 -----------------------
    # t1 = threading.Thread(target=increment_with_lock)
    # t2 = threading.Thread(target=decrement_with_lock)
    # t3 = threading.Thread(target=increment_without_lock)
    # t4 = threading.Thread(target=decrement_without_lock)
    # t1.start(), t2.start(), t1.join(), t2.join()
    # print("------------shared_resource_with_lock: ", shared_resource_with_lock, "-----------")
    # t3.start(), t4.start(), t3.join(), t4.join()
    # print("------------shared_resource_with_no_lock: ", shared_resource_without_lock, "-----------")


# ===============================================================
# ------------ 生产者-消费者 模型的 多种实现 ------------------------
# ===============================================================

# ---------- 多线程+队列 的生产者-消费者模型---------------------
# 使用子类继承的方式
# 使用的是 queue.Queue 这个线程安全的队列
# 当然，多线程也可以使用 multiprocess.Queue
class Producer(threading.Thread):
    def __init__(self, name, queue):
        super().__init__()
        self.name = name
        self.queue = queue

    def run(self):
        thread_id = threading.get_ident()
        pid = os.getpid()
        for i in range(10):
            item = random.randint(0, 256)
            # queue.Queue的put方法，block=True表示队列已满时会阻塞
            self.queue.put(item, block=True)
            print("thread '{}' in process '{}' is running".format(thread_id, pid))
            print('Producer notify: item {} is append to queue by {}'.format(item, self.name))
            sleep(0.5)


class Consumer(threading.Thread):
    def __init__(self, name, queue):
        super().__init__()
        self.name = name
        self.queue = queue

    def run(self):
        thread_id = threading.get_ident()
        pid = os.getpid()
        while True:
            # queue.Queue的get方法，block=True表示队列为空时会阻塞
            item = self.queue.get(block=True, timeout=5)
            print("thread '{}' in process '{}' is running".format(thread_id, pid))
            print("Consumer notify: item {} is popped from queue by {}".format(item, self.name))
            # queue.Queue的task_done() 方法用于通知队列已处理一个任务
            # self.queue.task_done()


# ----------- 多进程+队列 的生产者-消费者模型--------------------
# 这里没有使用子类继承的方式
# 它使用的 multiprocess.Queue 这个进程安全的队列
def producer(name, queue):
    print("producer " + name + " is running")
    pid = os.getpid()
    i = 0
    while i < 20:
        item = random.randint(0, 50)
        queue.put(item)
        print("process '{}' is running".format(pid))
        print("producer {} putting item {} successfully".format(name, item))
        i = i+1
        sleep(0.5)


def consumer(name, queue):
    pid = os.getpid()
    while True:
        # print("consumer {} get Queue size : {}".format(name, queue.qsize()))
        # item = queue.get()
        item = queue.get(block=True, timeout=5)
        print("process '{}' is running".format(pid))
        print("consumer {} getting item {} successfully.".format(name, item))
        sleep(0.5)


# if __name__ == "__main__":
    # --------多线程+队列 的 生产者-消费者 模型 --------------
    # 这里的TQueue 是 qtueue.Queue，它是线程安全的队列数据结构
    # queue = TQueue(3)
    # t1 = Producer('Producer-1', queue)
    # t2 = Consumer('Consumer-1', queue)
    # t3 = Consumer('Consumer-2', queue)
    # t1.start(), t2.start(), t3.start()
    # t1.join(), t2.join(), t3.join()

    # --------多进程+队列 的 生产者-消费者 模型--------------
    # queue = MQueue()
    # p = Process(target=producer, args=('Producer-1', queue))
    # c1 = Process(target=consumer, args=('Consumer-1', queue))
    # c2 = Process(target=consumer, args=('Consumer-2', queue))
    # p.start(), c1.start(), c2.start()
    # p.join(), c1.join(), c2.join()

    # ---- 将 multiprocess.Queue 用于多线程 -----------
    # queue = MQueue(3)
    # t1 = Producer('Producer-1', queue)
    # t2 = Consumer('Consumer-1', queue)
    # t3 = Consumer('Consumer-2', queue)
    # t1.start(), t2.start(), t3.start()
    # t1.join(), t2.join(), t3.join()


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


# if __name__ == "__main__":
    # ----- 线程同步：信号量 实现的 消费者-生产者 模型 ----------------
    # p = threading.Thread(target=producer_sem, args=('p-1', ))
    # c1 = threading.Thread(target=consumer_sem, args=('c-1', ))
    # c2 = threading.Thread(target=consumer_sem, args=('c-2', ))
    # p.start(), c1.start(), c2.start()
    # p.join(), c1.join(), c2.join()
    # print("--------------finished--------------")

    # ----- 线程同步：条件变量 实现的 消费者-生产者 模型 -----------------
    # p = Producer_cond()
    # c = Consumer_cond()
    # p.start(), c.start()
    # p.join(), c.join()

