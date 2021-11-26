import sys
import os


# --------- 类的定义和使用 --------------------
def __Class_Practice():
    pass

# 定义类
class Person:
    address = 'China'
    def __init__(self,name,sex,weight,height):
        self.name = name
        self._sex = sex
        self.__weight = weight
        self.__height__ = height
    def get_name(self):
        return self.name
    @classmethod
    def get_address(cls):
        return cls.address


# p1 = Person('Daniel', 'male', 65, 170)
# p1.name
# p1._sex
# p1.__weight
# p1.__height__
# p1.get_name()
# p1.address
# Person.address
# p1.address = "Earth"
# p1.address
# Person.address
# Person.address = 'Earth'
# p2 = Person('Daniel', 'male', 65 ,170)


# -------继承-----------------
class Student(Person):
    def __init__(self,name,sex,weight,sid):
        super().__init__(name,sex,weight)
        self.sid = sid
        
        
# s1 = Student('Daniel', 'male', 65, 20161212)


# ----------------装饰器----------------------------
def __Decorator_Practice():
    pass


def fun_decorator(f):
    def wrapper(*args, **kwargs):
        print("call:", f.__name__)
        print("positional arguments:", args)
        print("keyword arguments:", kwargs)
        return f(*args, **kwargs)
    return wrapper


def add(x, y):
    return x+y


# add(3,4)
# add_document = fun_decorator(add)
# add_document(3,4)


@fun_decorator
def add(x, y):
    return x+y

# add(3,4)


# 带参装饰器
def fun_decorator(text):
    def wrapper(f):
        def inner_wrapper(*args,**kwargs):
            print(text,f.__name__)
            print("positional arguments:",args)
            print("keyword arguments:",kwargs)
            return f(*args,**kwargs)
        return inner_wrapper
    return wrapper


@fun_decorator('execute')
def add(x, y):
    return x+y

# add(3,4)


# ---------- 协程 ----------------------------
def __Coroutine_Practice():
    pass

# yield 的生成器用法
def gen_fun():
    # 通常 yield 会在一个循环里，不过不是必须的
    print('start')
    # 注意，此时 yield 后面跟了返回的值，左边是没有表达式的
    yield 'A'
    print('continue')
    yield 'B'
    print('end.')

# if __name__ == '__main__':
#     for v in gen_fun():
#         print('--->', v)

# yield 的协程用法
def simple_coroutine():
    print('-> coroutine start')
    # 注意这里 yield 关键字的右边没有值，而左边有一个赋值表达式
    # 实际上，yield 有两种含义：1. 将yield右边的值发送给调用方；2.yield停止的地方可以接受调用方传来的数据，yield将该数据赋值给左边的表达式
    # 这里的 yield 后面没有值，只有左边有接收的表达式，说明此函数只接受数据，不产出数据
    x = yield
    print('-> coroutine received: ', x)

# if __name__ == '__main__':
#     my_coro = simple_coroutine()
#     print('my_coro: ', my_coro)
#     # 首先调用 next 方法，启动生成器，执行到 yield 处
#     next(my_coro)
#     # 执行到 yield 处之后，可以通过 send() 方法发送数据
#     my_coro.send(12)
#     # 这之后 yield 流程继续，直到最后抛出 StopIteration 异常

# 使用协程来实现计算均值
def coroutine_average():
    sum = 0.0
    count = 0
    average = None
    while True:
        value = yield average
        sum += value
        count += 1
        average = sum/count

cor_average = coroutine_average()
# 激活协程
next(cor_average)
# 计算均值
cor_average.send(2)
cor_average.send(3)
cor_average.send(4)
cor_average.send(5)
# 关闭协程
cor_average.close()