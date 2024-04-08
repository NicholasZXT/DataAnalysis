import inspect
import types
import time
import asyncio


# =============== 使用 asyncio 的前置准备 =========================
# 使用 asyncio 之前需要了解的一些内容

# 下面使用 yield 定义的是一个 生成器函数，不是生成器，也不是协程
def hello_generator(first_print, second_print):
    print(first_print)
    # 只要在函数里使用了 yield关键字，就是生成器函数，而且可以使用多次yield
    yield "hello_generator yield"
    # yield from 'ab'
    print(second_print)
    # 从 Python 3.3 开始，生成器函数末尾可以使用 return ，但是这个返回值实际上是放在抛出的 StopIteration 异常对象的 value 属性里
    return "hello_generator result"

def hello_gengerator_check():
    # 检查函数定义（没有调用之前）的对象
    print(hello_generator)
    # <function hello_generator at 0x0000029A69DD0CA0>
    print(type(hello_generator))
    # <class 'function'>
    # 不是 generator
    print(inspect.isgenerator(hello_generator))  # False
    # 也不是 协程
    print(inspect.iscoroutine(hello_generator))  # False
    # 而是 生成器函数(generatorfunction)
    print(inspect.isgeneratorfunction(hello_generator))  # True
    # 不是 协程函数(coroutinefunction)
    print(inspect.iscoroutinefunction(hello_generator))  # False

    # 检查函数调用之后的对象：调用生成器函数，会返回一个生成器对象
    t1 = hello_generator('first', 'second')
    # 调用之后，返回一个 generator
    print(t1)
    # <generator object hello_generator at 0x0000029A69DD4AC0>
    print(type(t1))
    # <class 'generator'>
    # 此时才是 generator
    print(inspect.isgenerator(t1))  # True
    # 但依旧不是协程
    print(inspect.iscoroutine(t1))  # False

    # 检查调用状态
    print(inspect.getgeneratorstate(t1))
    # GEN_CREATED
    # 生成器可以使用 next() 激活，也可以使用 send(None) 激活
    r = t1.send(None)
    # r 是 yield 后面交出的 "hello_generator yield"
    print(r)
    print(inspect.getgeneratorstate(t1))
    # GEN_SUSPENDED
    # 再次调用，会抛出 StopIteration 异常，并且 r 拿不到任何值
    # r = next(t1)
    # 除非自己处理该异常
    try:
        next(t1)  # 这里不会返回任何值，因为抛了异常
    except StopIteration as e:
        # hello_generator 的返回值是放在抛出的异常对象里
        r = e.value
    print(r)
    print(inspect.getgeneratorstate(t1))
    # GEN_CLOSED

    # 可以使用 for 循环来自动处理 迭代 和 StopIteration 异常，但是要注意，for循环获取的项目里，并不包含 return 后的返回值
    t2 = hello_generator('first', 'second')
    for item in t2:
        # 不会打印最后返回的 "hello_generator result"
        print(item)

# --------------------------------------------------------------

# 原生协程的定义，只需要用 async 即可，await不是必须的
async def hello_coroutine(first_print, second_print):
    print(first_print)
    time.sleep(1)
    print(second_print)
    # 协程的返回值有两种方式可以拿到：
    # 1. 使用 await 关键字调用协程，等待返回结果，这是常用方式 —— 这一点要特别关注
    # 2. 使用 StopIteration 异常获取，不常用
    return "hello_coroutine"

def hello_coroutine_check():
    # 检查协程定义
    # 未调用之前，是function，并且是 coroutinefunction 类型
    print(hello_coroutine)
    # <function hello_coroutine at 0x0000029A6738CCA0>
    print(type(hello_coroutine))
    # <class 'function'>
    # 不是协程
    print(inspect.iscoroutine(hello_coroutine))  # False
    # 是协程函数
    print(inspect.iscoroutinefunction(hello_coroutine))  # True

    # 检查调用
    # “调用”之后，返回的是协程类型，注意，返回的 t2 是协程对象，不是 hello_coroutine 的返回值 ----------------- KEY
    # 实际上，下面这种方式并不是真正调用协程函数，它只是创建了一个协程对象，必须放在下面的 await 关键字后面才是对协程函数的调用！！！
    t2 = hello_coroutine('first', 'second')
    print(t2)
    # <coroutine object hello_coroutine at 0x0000029A69D82040>
    print(type(t2))
    # <class 'coroutine'>
    print(inspect.iscoroutine(t2))  # True
    # 查看协程状态
    print(inspect.getcoroutinestate(t2))
    # CORO_CREATED

    # 原生协程不可以使用 next 激活
    # next(t2)  # 抛异常 TypeError: 'coroutine' object is not an iterator
    # 可以调用 send(None) 激活，不过下面会抛 StopIteration: hello_coroutine 异常
    # r = t2.send(None)
    # t2.close()
    # 和生成器函数一样，可以在 StopIteration 异常的 value 属性里获取协程返回值
    try:
        t2.send(None)
    except StopIteration as e:
        r = e.value
    print(r)


# --------------------------------------------------------------

# async + yield 定义的不是协程，而是 async_generator：异步生成器
# async + yield from 会抛出语法错误
async def hello_mix(first_print, second_print):
    print(first_print)
    yield 'hello_mix'
    # async 里不能使用yield from
    # yield from 'abc'
    print(second_print)
    # 异步生成器不能有返回值
    # return "hello_mix"


def hello_mix_check():
    print(hello_mix)
    # 类型仍然是 function
    print(type(hello_mix))
    # 不是协程
    print(inspect.iscoroutine(hello_mix))
    # 不是生成器
    print(inspect.isgenerator(hello_mix))
    # 不是异步生成器
    print(inspect.isasyncgen(hello_mix))
    # 是异步生成器函数
    print(inspect.isasyncgenfunction(hello_mix))
    # 不是下面两种类型的函数
    print(inspect.iscoroutinefunction(hello_mix))
    print(inspect.isgeneratorfunction(hello_mix))
    # 检查调用对象
    t3 = hello_mix('first', 'second')
    print(t3)
    # 类型是 async_generator
    print(type(t3))
    print(inspect.isasyncgen(t3))
    # 不是下面两种类型
    print(inspect.iscoroutine(t3))
    print(inspect.isgenerator(t3))
    # 没有可以检查异步生成器状态的方法
    print(inspect.getgeneratorstate(t3))
    print(inspect.getcoroutinestate(t3))
    # 异步生成器不可以使用 next 激活，也不可以用 .send(None) 激活
    next(t3)
    t3.send(None)
    # 这个东西不是很常用

# --------------------------------------------------------------

# 带有 await 的协程
# await 表示交出CPU的控制权：
# 1. 如果后面跟的是自定义的协程，那就是将执行权交给后面的协程；
# 2. 如果后面跟的是asyncio里的对象，那么就是将执行权交给asyncio的事件循环，由事件循环来将控制权交到下一个协程里
# 还有一个关键点：只有 await 关键词才能触发对协程的调用（还有对应的异常），拿到协程的返回值（包括异常） ------------------ KEY
async def hello_await(first_print, second_print):
    print(first_print)
    # await 后面的对象必须是 awaitable 的 —— 协程是 awaitable 对象
    res = await hello_coroutine('c1', 'c2')
    # hello_coroutine('c1', 'c2') 返回一个协程对象，await 关键字会调用这个协程对象并等待结果，最终拿到协程的返回值赋值给res ------- KEY
    print("res: ", res)

    # 如果使用 asyncio 的sleep() 方法，那就不能单独使用这个协程，必须要通过 asyncio 里的事件循环来驱动此协程
    # await asyncio.sleep(1)

    # 生成器 或者 异步生成器 都不是 awaitable 对象，不能放在 await 后面
    # await hello_generator('g1', 'g2')
    # await hello_mix('m1', 'm2')

    # 下面这个会循环调用，造成栈溢出
    # await hello_await('a1', 'a2')
    print(second_print)

def hello_await_check():
    print(hello_await)
    print(type(hello_await))
    print(inspect.iscoroutine(hello_await))
    print(inspect.iscoroutinefunction(hello_await))
    # 调用后，返回协程对象
    t4 = hello_await('first', 'second')
    print(t4)
    print(type(t4))
    print(inspect.iscoroutine(t4))
    print(inspect.getcoroutinestate(t4))
    t4.send(None)
    # 如果协程里使用了 asyncio.sleep(), 那就只能通过 asyncio 里的事件循环来驱动此协程，不能通过 .send(None) 来驱动
    t4 = hello_await('first', 'second')
    asyncio.run(t4)


# ======================= asyncio 的使用 ================================
# asyncio 所实现的异步编程有3个核心内容：1. 事件循环；2.协程；3.对协程对象的封装: Task对象
# 协程用于定义和封装需要执行代码，Task对象用于封装协程（它是Future对象的子类），驱动协程的执行，事件循环用于排定多个Task对象，在Task对象中转移控制权

# 定义一个协程，其中 await 了其他的协程，注意，这里面没有使用 asyncio 提供的任何函数
async def hello_asyncio():
    print(f"hello_asyncio start")
    # 使用 await 的地方，会在之后的异步函数执行开始之后，暂停当前函数的执行，等到其他异步函数执行完了，再继续执行——这和正常函数调用栈一样
    # 不同的地方在于，异步的调用只能保证顺序为 fun_1 > hello_coroutine('c1', 'c2')  > hello_coroutine('c3', 'c4')
    # 但是不能保证 hello_coroutine('c1', 'c2') 返回后马上继续执行 hello_coroutine('c3', 'c4')
    await hello_coroutine('c1', 'c2')
    await hello_coroutine('c3', 'c4')
    print(f"hello_asyncio end")

def hello_asyncio_run():
    f1 = hello_asyncio()
    print(inspect.iscoroutine(f1))
    # 自己手动执行激活协程，不过由于没有异常处理，最后返回的时候会抛出 StopIteration 异常
    f1.send(None)
    # 将该协程交给 asyncio 提供的事件循环执行，asyncio.run()会提供所需的事件循环+协程驱动+异常处理
    asyncio.run(f1)


# ------- 下面开始使用 asyncio 提供的一系列API 运行协程 --------
# 定义一个协程，内部使用 asyncio.sleep() 协程
async def say_after(delay, what):
    print(f"{what} <== at {time.strftime('%X')}.")
    # 这里的 asyncio.sleep()不同于 time.sleep()，它不是阻塞整个当前进程.
    # await 将控制权交给了 async.sleep() 协程之后，当前的协程就暂停在了这里，直到 async.sleep() 返回才继续执行
    # async.sleep()拿到控制权后，内部会做一些处理，比如设置一个回调函数，告诉事件循环 delay 秒之后再继续执行此处
    # 然后将控制权返回给 asyncio 里的事件循环，事件循环会驱动下一个协程执行
    await asyncio.sleep(delay)
    print(f"{what} ==> at {time.strftime('%X')}.")

def say_after_run():
    f2 = say_after(1, 'nothing')
    # 由于使用了 asyncio.sleep()，所以不能自己执行该协程，只能通过 asyncio 里的事件驱动来执行
    # f2.send(None)  # 这个会报错
    asyncio.run(f2)

# --------------------------
# 要想更清楚的了解 await 后面跟普通协程，和跟 asyncio提供的协程的区别，可以看下面的用例
# 重点是：直接 await 一个协程，并没有将它放到事件循环里进行执行 ------------------------- KEY
def show_current_tasks():
    """获取当前事件循环里未完成的Task，并打印出来"""
    tasks = [t.get_name() for t in asyncio.all_tasks()]
    print("running tasks in loop: ", tasks)

async def say_after_minus(delay, what):
    """递归调用函数"""
    # 此函数里没有调用 asyncio 的任何协程，所以不会将CPU控制权交还给 asyncio提供的事件循环
    print(f"{what} <== at {time.strftime('%X')}.")
    if delay > 1:
        # 每次递归，delay-1，然后 await 下一次递归的协程
        next_delay = delay - 1
        await say_after_minus(next_delay, "Task-sleep-"+str(next_delay))
    else:
        # 直到 delay=1 时，打印当前事件循环里的所有Task，检查前面几次递归的 say_after_minus 是否出现在事件循环里 ----- KEY
        show_current_tasks()
    print(f"{what} ==> at {time.strftime('%X')}.")

async def entry():
    # 用 entry 再进行一次封装，看看这个 entry 会不会被放到事件循环里
    await say_after_minus(4, 'Task-sleep-4')

async def main_loop_check():
    # 首先直接创建两个Task，它们肯定会被放入事件循环里
    t1 = asyncio.create_task(say_after(4, "t1"), name='t1')
    t2 = asyncio.create_task(say_after(4, "t2"), name='t2')
    # 然后 await 自定义协程，在最里层的递归返回处，获取当前事件循环里的所有Task
    await entry()
    # 下面的两个 await 可以不用写，不写的话，t1, t2可能执行不完
    await t1
    await t2

# asyncio.run(main_loop_check())

# 上面的调试结果如下：
# Task-sleep-4 <== at 11:33:36.
# Task-sleep-3 <== at 11:33:36.
# Task-sleep-2 <== at 11:33:36.
# Task-sleep-1 <== at 11:33:36.
# running tasks in loop:  ['Task-1', 't1', 't2']
# Task-sleep-1 ==> at 11:33:36.
# Task-sleep-2 ==> at 11:33:36.
# Task-sleep-3 ==> at 11:33:36.
# Task-sleep-4 ==> at 11:33:36.
# t1 <== at 11:33:36.
# t2 <== at 11:33:36.
# t1 ==> at 11:33:40.
# t2 ==> at 11:33:40.
# 可以看出，在递归返回处，获取事件循环里所有Task时，只拿到了3个Task，其中Task-1应该是 main_loop_check()，
# 并没有显示 entry(), say_after_minus() 的多次递归，说明这些自定义协程并没有放入事件循环里，
# await 将CPU控制权交给了这些自定义协程，没有交给asyncio的事件循环

async def main_loop_check_v2():
    t1 = asyncio.create_task(say_after(4, "t1"), name='t1')
    t2 = asyncio.create_task(say_after(4, "t2"), name='t2')
    # t2 = asyncio.create_task(say_after(5, "t2"), name='t2')
    # 调整下 await 的顺序
    await t1
    await entry()
    await t2

# asyncio.run(main_loop_check_v2())
# 运行结果如下
# t1 <== at 11:54:40.
# t2 <== at 11:54:40.
# t1 ==> at 11:54:44.
# t2 ==> at 11:54:44.
# Task-sleep-4 <== at 11:54:44.
# Task-sleep-3 <== at 11:54:44.
# Task-sleep-2 <== at 11:54:44.
# Task-sleep-1 <== at 11:54:44.
# running tasks in loop:  ['Task-1']
# Task-sleep-1 ==> at 11:54:44.
# Task-sleep-2 ==> at 11:54:44.
# Task-sleep-3 ==> at 11:54:44.
# Task-sleep-4 ==> at 11:54:44.
# 如果调整了 await 的顺序，则会发现，await t1 之后，t2也执行完了，所以拿到的task只有一个；
# 可能的猜想：await t1 时，CPU控制权交还给事件循环，它执行到 t1里的 await 之后，转而去执行事件循环里的t2，等到 t2 的await，转向执行 t1
# 剩余部分（此时 t2 可能执行完，也可能没执行完），然后控制权交到 entry() 这个自定义协程，等这个自定义协程执行完。


# -----------------------------------------------------------------------------------
# 下面的 5 个 main 函数，展示了asyncio基于事件循环的一些运行差异
async def main1():
    """
    这里的异步执行结果和同步执行是一样的，具体原因如下：
    此处没有向asyncio的事件循环里注册任何协程，所以事件循环里只有一个协程——通过asyncio.run(main1())提交的 main1 协程，
    运行到每个 await 处时，会将控制权转移给下一个协程 say_after，say_after协程本身不在asyncio的事件循环里，
    虽然 say_after 内部 await 了 asyncio.sleep()，将控制权交给了事件循环，但事件循环里处理时，找不到下一个可以执行的协程，
    仅有的 main1()协程需要等 await 返回结果，于是就等待1秒后继续执行 asyncio.sleep()，返回结果，执行完成 say_after，再继续执行 main1() 协程.
    遇到第 2 个 sayafter 协程时，重复上面的步骤，整个过程虽然是异步，但是结果显示和同步执行是一样的。
    """
    # 如果不使用 await，下面这句会抛出 RuntimeWarning: coroutine 'say_after' was never awaited
    # 理由就如之前所述，下面这样“调用”对于协程来说，并不是真正的调用执行，它只是返回一个协程对象，并不会执行其中的代码，只有在 await 后面才是真正的调用执行
    # say_after(3, 'No await')
    print(f"started at {time.strftime('%X')}.")
    # 这里通过 await 将控制权交给另一个协程
    await say_after(1, 'task1')
    print(f"middle at {time.strftime('%X')}.")
    await say_after(2, 'task2')
    print(f"finished at {time.strftime('%X')}.")

# 要想 并行 执行协程，需要将其注册为 asyncio事件循环的一个 Task 对象，如下所示：
async def main2():
    """
    此处展示的是 asyncio 的常规使用方式，有两个要点：
    1. 将每个协程先注册成事件循环里的Task对象
    2. 后续 await 对应的Task对象
    之后使用 asyncio.run(main2()) 之后，就能看到 task1, task2 对应的协程是异步执行的了，并且这里一定是 task1 和 task2 执行完毕后，
    再执行 main2 的收尾工作 —— 因为 main2 里 分别 await 了 task1, task2，但是 task1 和 task2 之间并没有 await 的关系，所以 task1
    和 task2 是异步并行执行的
    """
    print(f"started at {time.strftime('%X')}")
    # 使用 Task 包装协程，并使用 await —— 这是常规的使用方式
    task1 = asyncio.create_task(say_after(1, 'task1'))
    task2 = asyncio.create_task(say_after(2, 'task2'))
    # Wait until both tasks are completed (should take around 2 seconds.)
    await task1
    print(f"middle at {time.strftime('%X')}.")
    await task2
    print(f"finished at {time.strftime('%X')}")


# main2() 展示了 asyncio 里协程的常规用法，但有两个问题需要探究：
# 1. 将协程注册成 Task 到底意味着什么
# 2. await 之后的流程是怎样的
# 下面的几个 main 函数，用于研究这两个问题
async def main3():
    """
    本函数里，只是将协程封装成 Task 对象，后续并没有使用 await.
    调用 asyncio.run(main3()) 的结果显示：即使没有 await，task1和task2对应的协程也执行了，但是没有执行完毕.
    原因如下：
    调用 asyncio.create_task() 为协程创建Task对象时，也将该协程加入了事件循环里. main3() 里没有 await，表示此协程执行过程中不会交出控制权，
    而是一口气执行完；
    等到 main3() 协程执行完了，事件循环就执行下一个协程，也就是 task1，task1执行时 await，交出控制权，事件循环就执行task2，
    task2 也有 await，再次交出控制权，此时应该是轮到 task1 的 await 返回了；
    但是 由于此时 main3() 协程已经执行完了，事件循环就退出了，即使此时 task1 和 task2 还处于执行过程中
    """
    print(f"started at {time.strftime('%X')}")
    # 使用Task对象包装协程，但是没有 await，这里创建封装协程的Task的时候，就已经将该协程排进了事件循环里了
    task1 = asyncio.create_task(say_after(1, 'task1'))
    task2 = asyncio.create_task(say_after(2, 'task2'))
    print(f"middle at {time.strftime('%X')}.")
    print(f"finished at {time.strftime('%X')}")
    # 可以通过这两句来检查此时 task1 和 task2 的执行状态
    # print(task1.done())
    # print(task2.done())

async def main4():
    """
    此协程在 main3() 上继续研究，此协程里创建了 task，也没有使用 await 等待 task1和task2，而是通过分别下面两种方式交出控制权：
    1. 使用 time.sleep()：结果显示 main4 协程执行完之后才执行 task1和task2，没有和 main4 并发执行，并且task1和task2没有执行完.
    2. 使用 await asyncio.sleep()：结果显示 main4 协程和 task1、task2 是并发执行的，但是 task1 和 task2 能否执行完，要看 main4 里 sleep 的时间.
    原因如下：
    1. 所有的协程都是在同一个线程中执行的，使用 time.sleep() 会阻塞当前线程，不仅阻塞了 main4 协程本身，也阻塞了事件循环。等到 sleep() 返回之后，
    情况和 main3 是一样的。
    2. 通过 asyncio.sleep() 不是阻塞线程，而是将控制权交给事件循环，此时 事件循环里的 task1 和 task2 就有机会运行，所以可以看到 task1、task2是并发执行的。
    但是它们能否运行完，要看 main4() 协程 sleep() 的时间，如果 main4 sleep的时间不够 task1或task2 执行完，那么等到 main4 结束后，
    整个事件循环结束，它们也未能完成执行
    """
    print(f"started at {time.strftime('%X')}")
    # 同样的，使用Task对象包装协程，没有 await
    task1 = asyncio.create_task(say_after(1, 'task1'))
    task2 = asyncio.create_task(say_after(2, 'task2'))
    print(f"middle at {time.strftime('%X')}.")
    # 但分别通过两种 sleep 的方式交出控制权
    # time.sleep(1)
    await asyncio.sleep(3)   # 这里的时间决定了 task1 和 task2 能否执行完
    print(f"finished at {time.strftime('%X')}")
    # 可以通过这两句来检查此时 task1 和 task2 的执行状态
    # print(task1.done())
    # print(task2.done())

async def main5():
    """
    解决了 asyncio.create_task() 的疑惑，还需要研究一下 Task 的创建顺序对 await 的影响。
    这里按照 Task3, Task1, Task2 的顺序创建了 3 个协程，但是只 await 了其中 2 个 task
    结果：await task1 的时候，3个task就同时执行了，并且开始执行的顺序和创建的顺序是一样的；此外，task2 能否执行完，要看 await task3的执行时间.
    原因：
    1. 事件循环里task执行顺序取决于它们创建时加入的顺序，而不是await的顺序；
    2. 即使只 await 了 task1，也会执行其他的task，其中 排在 task1 之前的 task 一定会执行，排在 task1 之后的task，要看 task1 结束之前它们有没有放入事件循环，
    如果已经在事件循环里了，task1 执行过程中交出了控制权，之后的 task 也会执行
    3. 后续执行的 await task3，如果此时 task3 已经执行完，就不会交出控制权，直接拿到执行的结果，否则控制权会交回给事件循环
    4. 此外，依据task3的执行时间，来决定能否执行完 task2：如果await task3 时，如果 task2还没执行完，后续由于 main5 协程结束，导致时间循环结束，
    task2 也就没法执行完了。
    """
    print(f"started at {time.strftime('%X')}")
    # 使用 Task 包装协程，并使用 await，但是只 await 了一个 Task
    task3 = asyncio.create_task(say_after(3, 'task3'))
    task1 = asyncio.create_task(say_after(1, 'task1'))
    # task2 设置不同的时间，依据执行时间，task2有可能无法执行完毕
    # task2 = asyncio.create_task(say_after(2, 'task2'))
    task2 = asyncio.create_task(say_after(4, 'task2'))
    await task1
    print(f"middle at {time.strftime('%X')}.")
    print(task3.done())
    await task3
    print(f"finished at {time.strftime('%X')}")

# 测试上述的 main 函数
def run_main():
    asyncio.run(main1())
    asyncio.run(main2())
    asyncio.run(main3())
    asyncio.run(main4())
    asyncio.run(main5())

# run_main()

# -------------------------------------------------------------------
# 下面的这个例子，用来展示 await 下，异步编程的执行逻辑和顺序编程的执行逻辑的区别
# 创建两个子任务，其中分别运行一些协程
async def sub1():
    print(f"sub1 started at {time.strftime('%X')}.")
    task1 = asyncio.create_task(say_after(1, 'task1'))
    task3 = asyncio.create_task(say_after(2, 'task3'))
    await task1
    print(f"sub1 finished at {time.strftime('%X')}.")
    # 注意，这里没有 await task3， 而是直接返回了
    return task3

async def sub2():
    print(f"sub2 started at {time.strftime('%X')}.")
    task2 = asyncio.create_task(say_after(2, 'task2'))
    task4 = asyncio.create_task(say_after(1, 'task4'))
    # 这里通过 await 将控制权交给另一个协程
    await task2
    print(f"sub2 finished at {time.strftime('%X')}.")
    return task4

# 使用这个main函数主要是为了并行调用这两个协程内部的操作，因为 asyncio.run() 只能接受一个入口函数
async def main_parallel():
    """
    观察输出的顺序，会发现：
    1. main_parallel 里，先后 await 了 sub1 和 sub2，那么就一定能保证在 sub1 和 sub2 执行完之后，再执行 main_parallel 的收尾工作
    2. sub1 里，先后将 task1 和 task3 加入了事件循环，但是只 await task1，那么只能保证 task1 执行结束后再执行 sub1 的收尾工作，不能保证
       task3 在 sub1 内部结束 —— 实际上，task3 的结束是在 sub2 开始之后
    3. 异步编程的 await 最大的影响是改变了 **函数调用栈** 的情况 ------------------------------------ KEY
       以 sub1 为例，通常的函数调用栈下，应当是 sub1入栈 -> task1入栈 -> task1出栈 -> task3入栈 -> task3 出栈 -> sub1出栈 这个顺序，
       并且一定是 **连续的顺序**，但是在异步编程里，只能保证 [sub1入栈, task1入栈, task1出栈, sub1出栈] 这个大致顺序，不能保证连续执行，
       也不保证 task1出栈 一定在 task3入栈 之前；而且由于没有 await task3，所以 task3 的出栈不一定在 sub1出栈之前.
    4. sub2 的执行流程里，插入了 task3结束的部分，并且由于 sub2 也只是创建了 task4，没有 await task4，task4 的执行结束并不一定在 sub2
        结束之前，要看 task4 的 asyncio.sleep(delay) 时间设置
    5. 两个分隔线的打印顺序也不是预期的那样，分隔了 task3 和 task4 的打印结果，这也是因为 await 之前，sub1 里的 task3 和 sub2 里的 task4
       已经执行了，并且在 main_parallel 里的 await task3 之前就执行完了 —— 这说明多个 await 之间（包括他们之间平级的代码）是平行的关系
    """
    task5 = asyncio.create_task(say_after(1, 'task5'))
    task3 = await sub1()
    task4 = await sub2()
    await task3
    print("---------------------")
    await task4
    print("---------------------")
    await task5
    # 上面 task5 可以改为下面一句，但是这样的话，就会看到 task5 是最后执行的，并且执行顺序没有被中断，
    # 不像上面调用 asyncio.create_task 那样在创建 Task 对象的时候就开始执行了
    # await say_after(1, 'task5')

# asyncio.run(main_parallel())


# ---------- asyncio.run() 底层的操作 -------------
def asyncio_deep():
    # asyncio.run(main2()) 对应的底层操作如下（在交互式里无法运行，除非使用IPython）
    loop = asyncio.get_event_loop()
    task = loop.create_task(main2())
    # 上面两句也是 asyncio.create_task() 内部的操作
    loop.run_until_complete(task)
    pending = asyncio.all_tasks(loop=loop)
    for task in pending:
        task.cancel()
    group = asyncio.gather(*pending, return_exceptions=True)
    loop.run_until_complete(group)
    loop.close()


# ---------------------------------------------------------
# 异步编程需要注意的使用事项
async def fun_return():
    result = {'result': 'test result'}
    # 这里必须要加上 await
    await asyncio.sleep(1)
    return result

async def fun_return_check():
    # 不加 await 调用fun 时，得到的是一个 协程对象，不是该函数的返回结果 ！！！！
    # 下面这句会引发 RuntimeWarning: coroutine 'fun' was never awaited
    res1 = fun_return()
    print("res1.__class__: ", type(res1))
    print("res1: ", res1)
    # 只有 await + 调用fun 时，得到的才是函数的返回结果
    res2 = await fun_return()
    print("res2.__class__: ", type(res2))
    print("res2: ", res2)

# asyncio.run(fun_return_check())
