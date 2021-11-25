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
# 多进程间的管理器
from multiprocessing import Manager
from multiprocessing.managers import BaseManager, SyncManager, Namespace, BaseProxy, DictProxy
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
def __Serialization():
    pass

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
def __Pool_Serialization():
    pass

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


# ======================== Manager 使用 =======================================
def __Manager_Practice():
    pass

def worker(dict_proxy, list_proxy, key, item):
    dict_proxy[key] = item
    list_proxy.append(key)

# -------------- 自定义管理器 ----------------------
# 1. 自定义管理器必须继承于 BaseManager
class MyManager(BaseManager):
    # 类的定义体中什么都不需要写
    pass

# 2. 需要向自定义管理器中添加的共享数据对象，为 callable 对象，可以是类或者函数，甚至是匿名函数
class MathsClass:
    def __init__(self, x, y):
        self._x = x
        self._y = y
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    def set(self, x, y):
        self._x = x
        self._y = y
    def add(self):
        return self.x + self.y
    def __repr__(self):
        return f'(x: {self._x}, y: {self._y})'

def num_dict_fun():
    # 此函数返回 dict 作为共享对象
    num_dict = dict()
    return num_dict

# 3. 将上述共享对象类型注册到自定义的管理器中
MyManager.register('Maths', MathsClass)
# num_dict_fun 返回的共享对象是 dict，为了能正常使用，需要手动指定 DictProxy 代理类，这样才能使用 [] 访问符
MyManager.register('NumDict', num_dict_fun, DictProxy)
# MyManager.register('NumDict', num_dict_fun)  # 不指定代理类的话，[] 访问符就用不了

# 子进程执行的函数
def worker_fun(maths, num_dict, i):
    print(f'process {i}: math -- {maths}, dict -- {num_dict}')
    maths.set(i, i*2)
    print(f'process {i}: math -- {maths}')
    print(f'process {i}: math.add -- {maths.add()}')
    # 下面使用字典时需要注意，如果 register() 方法里没有指定代理对象，那么只能使用 setdefault 方法，因为自动生成的代理对象不会代理 特殊方法
    # 而 [] 访问符使用的是 __setitem__ 或者 __getitem__ 方法
    # 如果想正常使用字典，需要手动指定一个 DictProxy 代理对象
    num_dict[str(i)] = i*2
    # num_dict.setdefault(str(i), i*2)
    print(f'process {i}: dict -- {num_dict}')


if __name__ == '__main__':
    # 第 1 种使用方式，也是最简单的使用方式：使用已有的 SyncManager对象
    # ------ 注意，这种方式必须要是执行中，不能是被导入----
    # SyncManager 对象通常由顶层函数 Manager() 返回，不要手动创建
    # with Manager() as manager:
    #     # SyncManager 的监听地址
    #     print('manager.address: ', manager.address)
    #     # 创建列表和字典的代理对象
    #     l = manager.list()
    #     d = manager.dict()
    #     # 起 3 个进程，每个进程都要操作上面的两个代理对象
    #     # ------- 注意，代理对象一定可以被序列化，它可以在多个进程间传递 --------
    #     proc_list = [Process(target=worker, args=(d, l, i, i*2)) for i in [1, 2, 3]]
    #     for p in proc_list:
    #         p.start()
    #     for p in proc_list:
    #         p.join()
    #     print(l)
    #     print(d)

    # ------- 第 2 种，自定义管理器 ---------------
    # with MyManager() as manager:
    #     maths = manager.Maths(0, 0)
    #     num_dict = manager.NumDict()
    #     print(f'math.class: {maths.__class__}, math: {maths}')
    #     print(f'num_dict.class: {num_dict.__class__}, num_dict: {num_dict}')
    #     # 在下面的两个子进程中使用上述两个自定义共享对象的代理
    #     proc_list = [Process(target=worker_fun, args=(maths, num_dict, i)) for i in [1, 2]]
    #     for p in proc_list:
    #         p.start()
    #     for p in proc_list:
    #         p.join()
    #     print(f'math: {maths}')
    #     print(f'num_dict: {num_dict}')
    #     num_dict['a'] = 1
    #     # num_dict.setdefault('a', 1)
    #     print(f'num_dict: {num_dict}')

    # ------ 第 3 种，通过远程方式使用管理器 ---------------------
    manager = MyManager(address=('', 50000), authkey=b'abc')
    m_server = manager.get_server()
    print(f'm_server.address: {m_server.address}')
    # 启动服务进程，会阻塞
    m_server.serve_forever()
    # 对应的客户端见 multiprocess_basic.py
