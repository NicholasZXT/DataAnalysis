import inspect
import types
import time
import asyncio


# =============== 使用 asyncio 的前置准备 =========================
# 使用 asyncio 之前需要了解的一些内容

# 生成器 不是协程
def hello_generator(first_print, second_print):
    print(first_print)
    yield
    # yield from 'ab'
    print(second_print)


print(hello_generator)
print(inspect.isgenerator(hello_generator))
print(inspect.iscoroutine(hello_generator))
print(inspect.isgeneratorfunction(hello_generator))
print(inspect.iscoroutinefunction(hello_generator))
t1 = hello_generator('first', 'second')
print(t1)
print(inspect.isgenerator(t1))
print(inspect.iscoroutine(t1))
print(inspect.getgeneratorstate(t1))
t1.send(None)
print(inspect.getgeneratorstate(t1))
next(t1)
print(inspect.getgeneratorstate(t1))


# 原生协程的定义，只需要用 async 即可，await不是必须的
async def hello_coroutine(first_print, second_print):
    print(first_print)
    time.sleep(1)
    print(second_print)


print(hello_coroutine)
print(inspect.iscoroutine(hello_coroutine))
print(inspect.iscoroutinefunction(hello_coroutine))
t2 = hello_coroutine('first', 'second')
print(t2)
print(inspect.iscoroutine(t2))
print(inspect.getcoroutinestate(t2))
# 原生协程不可以使用 next 激活，可以调用 send(None) 激活
next(t2)
t2.send(None)


# async + yield 定义的不是协程，而是 async_generator：异步生成器
# async + yield from 会抛出语法错误
async def hello_mix(first_print, second_print):
    print(first_print)
    yield 'hello_mix'
    # async 里不能使用yield from
    # yield from 'abc'
    print(second_print)

print(hello_mix)
print(inspect.iscoroutine(hello_mix))
print(inspect.isgenerator(hello_mix))
print(inspect.isasyncgen(hello_mix))
print(inspect.iscoroutinefunction(hello_mix))
print(inspect.isgeneratorfunction(hello_mix))
print(inspect.isasyncgenfunction(hello_mix))
t3 = hello_mix('first', 'second')
print(t3)
print(inspect.iscoroutine(t3))
print(inspect.isgenerator(t3))
print(inspect.isasyncgen(t3))
# 没有可以检查异步生成器状态的方法
print(inspect.getgeneratorstate(t3))
print(inspect.getcoroutinestate(t3))
# 异步生成器不可以使用 next 激活，也不可以用 .send(None) 激活
next(t3)
t3.send(None)
# 这个东西不是很常用


# 带有 await 的协程
# await 表示交出CPU的控制权：
# 如果后面跟的是自定义的协程，那就是将执行权交给后面的协程；
# 如果后面跟的是asyncio里的对象，那么就是将执行权交给asyncio的事件循环，由事件循环来将控制权交到下一个协程里
async def hello_await(first_print, second_print):
    print(first_print)
    # await 后面的对象必须是 awaitable 的
    # 协程是 awaitable 对象
    await hello_coroutine('c1', 'c2')

    # 如果使用 asyncio 的sleep() 方法，那就不能单独使用这个协程，必须要通过 asyncio 里的事件循环来驱动此协程
    # await asyncio.sleep(1)

    # 生成器 或者 异步生成器 都不是 awaitable 对象，不能放在 await 后面
    # await hello_generator('g1', 'g2')
    # await hello_mix('m1', 'm2')

    # 下面这个会循环调用，造成栈溢出
    # await hello_await('a1', 'a2')
    print(second_print)


print(hello_await)
print(inspect.iscoroutine(hello_await))
print(inspect.iscoroutinefunction(hello_await))
t4 = hello_await('first', 'second')
print(t4)
print(inspect.iscoroutine(t4))
print(inspect.getcoroutinestate(t4))
t4.send(None)
# 如果协程里使用了 asyncio.sleep(), 那就只能通过 asyncio 里的事件循环来驱动此协程，不能通过 .send(None) 来驱动
t4 = hello_await('first', 'second')
asyncio.run(t4)


# ======================= asyncio 的使用 ================================
# asyncio 所实现的异步编程有3个核心内容：1. 事件循环；2.协程；3.对协程对象的封装: Future或者Task对象
# 协程用于定义和封装需要执行代码，Task对象或者Future对象用于封装协程，驱动协程的执行，事件循环用于排定多个Task对象，在Task对象中转移控制权

# 定义一个协程，其中 await 了其他的协程，注意，这里面没有使用 asyncio 提供的任何函数
async def fun_1():
    print(f"fun_1 start")
    await hello_coroutine('c1', 'c2')
    await hello_coroutine('c3', 'c4')
    print(f"fun_1 end")

f1 = fun_1()
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

f2 = say_after(1, 'nothing')
# 由于使用了 asyncio.sleep()，所以不能自己执行该协程，只能通过 asyncio 里的事件驱动来执行
# f2.send(None)  # 这个会报错
asyncio.run(f2)


# 下面的 5 个 main 函数，展示了asyncio基于事件循环的一些运行差异
async def main1():
    """
    这里的异步执行结果和同步执行是一样的，具体原因如下：
    此处没有向asyncio的事件循环里注册任何协程，所以事件循环里只有一个协程——通过asyncio.run(main1())提交的本协程，
    运行到每个 await 处时，会将控制权转移给下一个协程 say_after，say_after协程本身不在asyncio的事件循环里，
    虽然 say_after 内部 await 了 asyncio.sleep()，将控制权交给了事件循环，但事件循环里处理时，找不到下一个可以执行的协程，
    仅有的 main1()协程需要等 await 返回结果，于是就等待1秒后继续执行 asyncio.sleep()，返回结果，执行完成 say_after，再继续执行 main1() 协程.
    遇到第 2 个 sayafter 协程时，重复上面的步骤，整个过程虽然是异步，但是结果显示和同步执行是一样的。
    """
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
    之后使用 asyncio.run(main2()) 之后，就能看到 task1, task2 对应的协程是异步执行的了.
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
    # 使用Task对象包装协程，但是没有 await
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
asyncio.run(main1())
asyncio.run(main2())
asyncio.run(main3())
asyncio.run(main4())
asyncio.run(main5())


# ---------- asyncio.run() 底层的操作 -------------
# asyncio.run(main2()) 对应的底层操作如下（在交互式里无法运行）
loop = asyncio.get_event_loop()
task = loop.create_task(main2())
loop.run_until_complete(task)
pending = asyncio.all_tasks(loop=loop)
for task in pending:
    task.cancel()
group = asyncio.gather(*pending, return_exceptions=True)
loop.run_until_complete(group)
loop.close()