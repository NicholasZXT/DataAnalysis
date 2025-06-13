"""
Joblib Practice
官方文档 https://joblib.readthedocs.io/en/stable/
Joblib库的目的是提供 lightweight pipelining in Python，具体来说，就是提供了如下3个方面的功能：

（1）函数调用缓存，专门针对输入输出结果比较大（尤其是numpy）的函数的调用进行缓存。
这部分主要是如下类/函数：
- Memory 类
- MemorizedResult 类
- expires_after() 函数

（2）并行计算
- Parallel 类
- delayed() 函数
- cpu_count() 函数
- effective_n_jobs() 函数
- parallel_config()
- parallel_backend()

（3）快速序列化/反序列化
- dump() 函数
- load() 函数

"""
import os
import time
import math
import numpy as np
from joblib import Memory, expires_after, MemorizedResult
from joblib import Parallel, delayed, cpu_count, effective_n_jobs, parallel_config, parallel_backend
from joblib import dump, load


def memory_usage():
    """
    主要使用 Memory 类进行缓存。
    默认使用本地文件作为缓存，所以初始化时需要配置 location 参数
    提供了如下方法：
    - cache
    - clear
    - eval
    """
    store_dir = os.path.join(os.getcwd(), "joblib_cache")
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    memory = Memory(location=store_dir, backend="local", verbose=0)

    @memory.cache
    def f(x):
        print('Running f(%s)' % x)
        return x

    print(f(1))
    print(f(1))
    print(f(2))

    # 检查该函数的此次调用是否命中缓存
    print(f.check_call_in_cache(1))
    # 测试调用
    memory.eval(f, 1)
    # 清楚缓存
    memory.clear()

    # joblib还提供了一个expires_after()函数，用于设置缓存的过期时间
    @memory.cache(cache_validation_callback=expires_after(seconds=10))
    def f1(x):
        print('Running f(%s)' % x)
        return x


def parallel_usage():
    """
    并行计算主要使用 Parallel 类.
    delayed() 函数主要是为了提取函数调用的签名，是一个辅助工具。
    """

    def my_fun(num: int) -> float:
        time.sleep(1)
        return math.sqrt(num ** 2)

    iter_rounds = 5

    # 使用 for 循环
    start = time.time()
    for i in range(iter_rounds):
        my_fun(i)
    end = time.time()
    print('{:.4f} s'.format(end - start))

    # 使用 Parallel , 默认是多进程
    start = time.time()
    # 使用 2 个进程
    Parallel(n_jobs=2)(delayed(my_fun)(i) for i in range(iter_rounds))
    end = time.time()
    print('{:.4f} s'.format(end - start))

    # 对比可以看出，节省了差不多一半的时间

    # 其他工具函数
    print(cpu_count())


def serialization_usage():
    """
    主要是如下两个函数：
    - dump()
    - load
    相比于pickle，对于包含 numpy 数据的对象来说，比较好用。
    """
    ...


def main():
    memory_usage()
    parallel_usage()
    serialization_usage()


if __name__ == '__main__':
    main()
