import inspect
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
async def hello_mix(first_print, second_print):
    print(first_print)
    yield 'hello_mix'
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


async def fun_1():
    print(f"fun_1 start")
    await hello_coroutine('c1', 'c2')
    await hello_coroutine('c3', 'c4')
    print(f"fun_1 end")

f1 = fun_1()



async def say_after(delay, what):
    print(f"{what} <==")
    await asyncio.sleep(delay)
    print(f"{what} ==>")


async def main1():
    print(f"started at {time.strftime('%X')}")
    await say_after(1, 'hello')
    await say_after(2, 'world')
    print(f"finished at {time.strftime('%X')}")


async def main2():
    print(f"started at {time.strftime('%X')}")
    task1 = asyncio.create_task(say_after(1, 'task1'))
    task2 = asyncio.create_task(say_after(2, 'task2'))
    # Wait until both tasks are completed (should take
    # around 2 seconds.)
    await task1
    await task2
    # time.sleep(1)
    print(f"finished at {time.strftime('%X')}")



asyncio.run(main1())
# asyncio.run(main2())