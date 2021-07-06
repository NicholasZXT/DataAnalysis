"""
Python 并行编程
"""
import os
from time import sleep
import pickle
import random
# 多线程相关
from threading import Thread, get_ident
from queue import Queue as TQueue, Full, Empty
# 多进程相关
from multiprocessing import Process, Pool, Semaphore, Condition, Queue as MQueue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# 其他
from elasticsearch import Elasticsearch

# ================ ES 测试 ===============================
# es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# es.info()
# es.close()

# ================= 线程和进程的参数序列化 =============================
# 考察多线程中能否传入其他对象 ------------- 答案是可以，因为线程共享进程的空间，使用 es 对象时不需要序列化
class ThreadFunction(Thread):
    def __init__(self, name, es):
        super().__init__()
        self.name = name
        self.es = es

    # def run(self) -> None:
    #     super().run()
    def run(self):
        thread_id = get_ident()
        print("thread " + self.name + " with thread id ‘{}’ starting".format(thread_id))
        es_info = self.es.info()
        print("es_info:\n ", es_info)
        print("thread " + self.name + " with thread id ‘{}’ ending".format(thread_id))


def thread_fun(name, es):
    thread_id = get_ident()
    print("thread " + name + " with thread id ‘{}’ starting".format(thread_id))
    es_info = es.info()
    print("es_info:\n ", es_info)
    print("thread " + name + " with thread id ‘{}’ ending".format(thread_id))


# 考察多进程中能否传入其他对象 ----------- 答案是 不可以 ！！！ 因为 es 对象不能序列化
class ProcessFunction(Process):
    def __init__(self, name, es):
        super().__init__()
        self.name = name
        self.es = es

    # def run(self) -> None:
    #     super().run()
    def run(self):
        pid = os.getpid()
        print("process " + self.name + " with pid ‘{}’ starting".format(pid))
        es_info = self.es.info()
        print("es_info:\n ", es_info)
        print("process " + self.name + " with pid ‘{}’ ending".format(pid))


def process_fun(name, es):
    pid = os.getpid()
    print("process " + name + " with pid ‘{}’ starting".format(pid))
    es_info = es.info()
    print("es_info:\n ", es_info)
    print("process " + name + " with pid ‘{}’ ending".format(pid))


# if __name__ == '__main__':
#     print("main process start")
#     es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
#     print("-------------------thread----------------------------")
#     t1 = ThreadFunction('thread-class', es)
#     t1.start(), t1.join()
#     f1 = Thread(target=thread_fun, args=('thread-fun', es))
#     f1.start(), f1.join()
#     print("-------------------process---------------------------")
#     # es 对象不能序列化，所以无法传递给子进程，下面会报错
#     t2 = ProcessFunction('process-class', es)
#     t2.start(), t2.join()
#     f2 = Process(target=process_fun, args=('process-fun', es))
#     f2.start(), f2.join()
#     es.close()
#     print("main process end")


# ======================= 对象序列化 ===================================
# Person 类可以序列化
class Person:
    def __init__(self, name):
        self.name = name

    def print_name(self):
        print("Person name is :", self.name)

    def __repr__(self):
        return "Person.name : {}".format(self.name)

# 但是下面的 EsClient 类不可以序列化，因为它保存了外部连接的状态
class EsClient:
    def __init__(self, es_host, item=None):
        self.es_host = es_host
        self.es = Elasticsearch([{'host': es_host, 'port': 9200}])
        self.item = item

    def es_info(self):
        info = self.es.info()
        print("es_client info:\n", info)

    def __repr__(self):
        return "es_host:{}, item:{}".format(self.es_host, self.item)


# if __name__ == '__main__':
#     p = Person('python')
#     print("person repr: ", p)
#     p.print_name()
#     p_ser = pickle.dumps(p)
#     print("person dumps : ", p_ser)
#
#     es_host = 'localhost'
#     es_client = EsClient(es_host)
#     es_client.es_info()
#     # 下面的序列化会失败
#     es_ser = pickle.dumps(es_client)
#     print("es_client dumps: ", es_ser)


# ================= 进程池 + 队列的序列化问题 -- KEY ===============================
def producer(name, queue):
    pid = os.getpid()
    print("producer ’{}‘ with pid ’{}‘ start.".format(name, pid))
    i = 0
    while i < 20:
        item = random.randint(0, 50)
        # 放入队列的是 Person 类实例，它可以被序列化，能成功放入队列
        # item = Person(name=item)
        # 放入队列的是 EsClient 类实例，它不能被序列化，所以无法放入队列
        # item = EsClient(es_host='local', item=item)
        # queue.put(item, block=True, timeout=5)
        try:
            queue.put(item, block=True, timeout=5)
        except Full:
            print("producer ’{}‘ with pid ’{}‘ can't put item to Queue, Queue is full".format(name, pid))
            break
        print("producer ’{}‘ with pid ’{}‘ putting item: {}, Queue size : {}".format(name, pid, item, queue.qsize()))
        # print("producer ’{}‘ with pid ’{}‘ get Queue size : {}".format(name, pid, queue.qsize()))
        i += 1
        sleep(0.5)
    print("producer ’{}‘ with pid ’{}‘ is done".format(name, pid))
    return None


def consumer(name, queue):
    pid = os.getpid()
    print("consumer ’{}‘ with pid ’{}‘ start.".format(name, pid))
    while True:
        # print("consumer ’{}‘ with pid ’{}‘ get Queue size : {}".format(name, pid, queue.qsize()))
        # item = queue.get()
        # item = queue.get(block=True, timeout=5)
        try:
            item = queue.get(block=True, timeout=5)
        except Empty:
            print("consumer ’{}‘ with pid ’{}‘ can't get item from Queue, Queue is empty".format(name, pid))
        print("consumer ’{}‘ with pid ’{}‘ getting item: {}, Queue size : {}.".format(name, pid, item, queue.qsize()))
        sleep(0.5)


# if __name__ == '__main__':
    # 队列里放入的，也必须是可序列化的对象，比如 Person 类实例，但是 EsClient 类实例就不行
    # queue = MQueue(maxsize=10)
    # p = Process(target=producer, args=('Producer-1', queue))
    # c1 = Process(target=consumer, args=('Consumer-1', queue))
    # c2 = Process(target=consumer, args=('Consumer-2', queue))
    # p.start(), c1.start(), c2.start()
    # p.join(), c1.join(), c2.join()

    # 进程池 + 队列 ——  ProcessPoolExecutor 中不能传入 Queue 作为参数！！！
    # queue = MQueue(maxsize=10)
    # 单个生产者
    # p = Process(target=producer, args=('Producer-1', queue))
    # p.start()
    # print("main process")
    # 生产者进程池 往队列里放入任务 —— 一个主进程里，只能开启一个进程池（它会阻塞），所以这里不能开启 生产者进程池
    # future_producers = []
    # with ProcessPoolExecutor(max_workers=2) as executor:
    #     print("producer executor: ", executor)
    #     for i in range(2):
    #         print("submit producer: {}".format(i))
    #         executor.submit(producer, name='Producer-{}'.format(i), queue=queue)
    #         # future = executor.submit(producer, name='Producer-{}'.format(i), queue=queue)
    #         # future_producers.append(future)
    # print("main process")
    # 消费者进程池 从队列里消费任务
    # future_consumers = []
    # with ProcessPoolExecutor(max_workers=2) as executor:
    #     print("consumer executor: ", executor)
    #     for i in range(2):
    #         print("submit consumer: {}".format(i))
    #         # 传入参数为 Queue 时，不会报错，但是也不会运行
    #         executor.submit(consumer, name='Consumer-{}'.format(i), queue=queue)
            # future = executor.submit(consumer, name='Consumer-{}'.format(i), queue=queue)
            # future_consumers.append(future)
    # for future in future_producers:
    #     print("future_producers.result: ", future.result())
    # for future in future_consumers:
    #     print("future_consumers.result: ", future.result())


    # 线程池
    # future_list = []
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     for i in range(3):
    #         future = executor.submit(worker, "Thread",  i+1)
    #         future_list.append(future)
    # for future in future_list:
    #     print("future.result: ", future.result())