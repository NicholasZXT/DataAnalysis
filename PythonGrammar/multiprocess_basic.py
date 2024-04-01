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
from multiprocessing import Process, Pool, Semaphore, Condition, current_process
from multiprocessing import Queue as MQueue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Python里的多线程并发编程相比Java要简单很多，没有synchronize，volatile等关键字，主要原因（针对CPython实现）如下：
# 1. Python里面大部分的单次操作/方法都是原子性的，这里的原子性准确来说是该操作对应的是Python字节码的一行，再加上GIL的存在，所以不太需要 synchronize
# 2. Python里的读写操作，都是从主内存里读取的，同样也加上GIL的存在，所以多线程里不会出现CPU缓存不一致的情况，也就不需要 volatile 关键字
# 待解决的一个疑问是：Python里的多线程，是否会从头到尾都只使用CPU的同一个Core？ 我个人感觉答案应该是否

# =================== 基本的线程使用 ============================
# 方法一，传入线程里要执行的函数
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
#     # ---------基本线程使用---------------
#     # 创建线程
#     t1 = threading.Thread(target=my_fun, args=("thread-1",), name="thread-1")
#     t2 = ThreadFunction(num="thread-2")
#     # 开始线程
#     t1.start(), t2.start()
#     # 由于 GIL 的限制，如果主进程一直在执行，那么就不会释放 GIL ，导致该进程中的其他线程拿不到控制权  ----------- KEY
#     # 所以如果下面一直在执行 while 循环，到不了 join，那么子进程就一直不会执行
#     while True:
#         print('main thread running.')
#     # join表示 主线程 在此处阻塞，等待线程执行结束后再继续，只有主线程阻塞了，其他线程才能拿到CPU
#     t1.join(), t2.join()


# =================== 基本的进程使用 =============================
# 方法一，传入进程里要执行的函数, 函数同线程的 my_fun
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


# =================== 对象在进程间的传递 ============================
def __Object_Passing():
    pass

# 创建进程时，使用继承Process的方法比较容易理解，但是使用传入函数涉及到对象时，会碰到如下两个问题：
# 1. 某个对象可以将自身传入到另一个对象吗？            --- 可以
# 1. 传入进程的函数是某个对象的方法时，会发生什么情况？   --- 会将当前对象也传入到子进程中，所以当前对象必须要是可序列化的

class Company:
    def __init__(self, person, company_name):
        # person 是一个 Person 类实例, company_name 是 str
        # 如果在这里打断点，debug停在这里时：
        # 1. 调用 id(person) 会发现和外面的 person 是同一个实例，表明实例对象可以将自己传入到另一个对象中
        # 2. 调用 person.print_company() 时会触发 AttributeError，提示 company_info 属性不存在——因为这里的 Company 对象还没完成实例化，所以
        # 外面 person 的 company_info 属性此时还没有创建，所以调用 print_company() 方法访问不到该属性
        self.person = person
        self.company_name = company_name

    def print_company(self):
        print(f"{self.person} is a member of company {self.company_name}.")


class Person:
    def __init__(self, name, company_name=None):
        self.name = name
        if company_name:
            # 注意，这里传入 self 表示 当前Person类实例对象本身
            self.company_info = Company(self, company_name)
        else:
            self.company_info = None

    def print_name(self):
        # 这个方法会被传递到子进程中执行，通过打印的如下信息，会发现在传递 print_name 方法的同时，也会将 当前类的实例传递到 子进程中
        # 但是 Linux 和  Windows 下不一样的是：
        # Linux是通过 fork() 产生子进程，子进程继承父进程里的所有对象，因此这里的 id(self) 等同于父进程的 id
        # Windows下是通过 pickle 之后传入子进程的，因此这里的 id(self) 不同于父进程里的 id
        print('Process : {}, PID: {}, object id(self): {}'.format(current_process(), os.getpid(), id(self)))
        print("Person name is :", self.name)

    def print_company(self):
        if self.company_info:
            self.company_info.print_company()
        else:
            print(f"{self.name} is freedom.")

    def __repr__(self):
        return "Person.name : {}".format(self.name)


# if __name__ == '__main__':
#     # 初始化这个类的时候，检查一下 Company 类的实例过程
#     p = Person('Daniel', 'Empire')
#     # 开启子进程时，检查一下传入的内容
#     print('Process : {}, PID: {}, object id(p): {}'.format(current_process(), os.getpid(), id(p)))
#     proc = Process(target=p.print_name)
#     proc.start()
#     proc.join()


# ================== 单例模式 + 多进程 =========================
# 下面这个例子可以看出，单例模式的作用范围是 单进程，跨进程的话是可以有两个对象的，并且这两个对象的修改都是独立的
class Single:
  __instance = None
  # 这里不是严格的单例模式，因为这个构造方法没有被隐藏起来
  def __init__(self, data):
    self.data = data

  @classmethod
  def get_instance(cls, data):
    if cls.__instance is None:
      # print('creating instance')
      cls.__instance = Single(data)
    return cls.__instance

  def print_data(self):
      print(self.data)

def sub_proc(single):
    single.data = 'new + ' + single.data
    print(f"id(single): {id(single)}")
    single.print_data()
    single_2 = Single.get_instance('new-data-2')
    print(f"id(single_2): {id(single_2)}")
    single_2.print_data()


# if __name__ == '__main__':
#     data = 'singleton'
#     single = Single.get_instance(data)
#     print(f"id(single): {id(single)}")
#     single_2 = Single.get_instance(data)
#     print(f"id(single_2): {id(single_2)}")
#     single.print_data()
#     proc = Process(target=sub_proc, args=(single,))
#     proc.start()
#     proc.join()
#     single.print_data()


# ============================================================
# ----------------进程或线程池的使用-----------------------------
# ============================================================
def __Pool_Practice():
    pass

def worker(level, msg):
    print("{} of {} starting, process id is: {}.".format(level, msg, os.getpid()))
    # random.random()随机生成0~1之间的浮点数
    sleep_time = random.random()*5
    sleep(sleep_time)
    print("{} of {} sleep for {:.4f} second.".format(level, msg, sleep_time))
    return sleep_time

def show(msg):
    print("msg '{}' starting, process id is: {}.".format(msg, os.getpid()))
    # random.random()随机生成0~1之间的浮点数
    sleep_time = random.random()*5
    sleep(sleep_time)
    # print("msg '{}' sleep for {:.4f} second.".format(msg, sleep_time))
    # return sleep_time
    return msg

# if __name__ == "__main__":
    # -------- Pool 的使用------------------

    # 使用 apply 方法，一次提交一个进程，并阻塞直到子进程执行完
    # 这种方式不必使用 join 等待，返回的结果就是直接是函数的返回结果
    # with Pool(3) as pool:
    #     res = pool.apply(worker, ('process', 'worker-1'))
    #     # 返回值就是 worker方法 的返回值
    #     print("worker-1 is done, res is :", res)
    #     res = pool.apply(worker, ('process', 'worker-2'))
    #     print("worker-2 is done, res is :", res)
    #     res = pool.apply(worker, ('process', 'worker-3'))
    #     print("worker-3 is done, res is :", res)

    # 使用 apply_async 进行异步调用，返回的 res 是一个 pool.ApplyResult 对象
    # 必须要 使用 join 方法开启任务
    # pool = Pool(3)
    # res1 = pool.apply_async(worker, ('process', 'worker-1'))
    # print("worker-1 is done, res is: ", res1)
    # res2 = pool.apply_async(worker, ('process', 'worker-2'))
    # print("worker-2 is done, res is: ", res2)
    # res3 = pool.apply_async(worker, ('process', 'worker-3'))
    # print("worker-3 is done, res is: ", res3)
    # print(f"res1.__class__: {type(res1)}")
    # # 判断是否执行完成，它不会抛出异常
    # print(f"res1.ready(): {res1.ready()}")
    # # 但是下面的这个方法会抛出 ValueError 异常
    # # print(f"res1.successful(): {res1.successful()}")
    # print("----start----")
    # pool.close()  # 关闭进程池，关闭后po不再接收新的请求 —— 必须要在 .join() 前调用此方法
    # pool.join()   # 等待po中所有子进程执行完成，再执行下面的代码,可以设置超时时间join(timeout=)
    # print("-----end-----")
    # # 在执行完成后检查则不会抛出异常
    # print(f"res1.successful(): {res1.successful()}")
    # # wait() 会等待结果执行完成，可以设置 timeout 参数，它不会返回任何值
    # print(f"res1.wait(): {res1.wait()}")
    # # get() 用于获取结果，可以设置 timeout 参数
    # print(f"res1.get(): {res1.get()}")

    # 使用 map/starmap 方法一次提交多个进程，使用进程池中的所有进程并行执行某个函数 ----- 不太好用
    # map 只能给函数传一个参数，starmap 可以传入多个参数
    # pool = Pool(3)
    # # 这里由于 worker 有两个参数，所以要使用 starmap
    # # res = pool.starmap(worker, [('process', 'worker-1')])
    # res = pool.starmap(worker, [('process', 'worker-1'), ('process', 'worker-2'), ('process', 'worker-3')])
    # # 上面的方法会阻塞，直到进程池执行完毕
    # # 返回的 res 是一个 list，其中的值就是 worker 的返回值
    # print(res)
    # print("----start----")
    # pool.close()  # 关闭进程池，关闭后po不再接收新的请求
    # pool.join()  # 等待po中所有子进程执行完成，再执行下面的代码,可以设置超时时间join(timeout=)
    # print("-----end-----")

    # Pool.map 方法有一个 chunksize 参数，指定一次传一批数据到子进程了，而不是每次传一条
    # pool = Pool(2)
    # # res = pool.map(show, [1, 2, 3, 4, 5, 6, 7, 8])
    # res = pool.map(show, [1, 2, 3, 4, 5, 6, 7, 8], chunksize=4)
    # # 返回的结果顺序并不会乱
    # print(res)

    # --------- 使用  concurrent.futures ---------------------
    # 线程池
    # future_list = []
    # with ThreadPoolExecutor(max_workers=3) as executor:
    #     for i in range(3):
    #         future = executor.submit(worker, "Thread",  i+1)
    #         future_list.append(future)
    # for future in future_list:
    #     print("future.result: ", future.result())

    #  进程池
    # future_list = []
    # with ProcessPoolExecutor(max_workers=3) as executor:
    #     for i in range(3):
    #         future = executor.submit(worker, "Process",  i+1)
    #         future_list.append(future)
    # for future in future_list:
    #     print("future.result: ", future.result())


# ============================================================
# ---------------- 线程同步 -----------------------------------
# ============================================================
def __Lock_Practice():
    pass

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
        # += 这个操作不是原子性的
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
    # 下面的这个无锁冲突演示，实践中不那么容易成功
    # t3.start(), t4.start(), t3.join(), t4.join()
    # print("------------shared_resource_with_no_lock: ", shared_resource_without_lock, "-----------")


# ===============================================================
# ------------ 生产者-消费者 模型的 多种实现 ------------------------
# ===============================================================
def __Producer_Consumer_Practice():
    pass

# ---------- 多线程 + 线程队列 的生产者-消费者模型---------------------
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


# ----------- 多进程 + 进程队列 的生产者-消费者模型--------------------
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


# ======================== Manager 使用 =======================================
def __Manager_Client_Practice():
    pass

# 自定义Manager管理器，用作客户端
from multiprocessing.managers import BaseManager
# 进程里执行的函数
from PythonGrammar.multiprocess_ipc import worker_fun

class MyManagerClient(BaseManager):
    # 类的定义体中什么都不需要写
    pass

# 这里注册共享对象时，只需要提供共享数据类型的 typeid ——它们对应于远程Manager服务端的共享数据类型
# 由远程Manager服务端返回，所以这里不需要提供定义
MyManagerClient.register('Maths')
MyManagerClient.register('NumDict')

# if __name__ == '__main__':
#     manager_client = MyManagerClient(address=('localhost', 50000), authkey=b'abc')
#     # 调用这一句连接远程Manager服务
#     manager_client.connect()
#     maths = manager_client.Maths(0, 0)
#     num_dict = manager_client.NumDict()
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
