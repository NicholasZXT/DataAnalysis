import sys
import os


# --------- 类的定义和使用 --------------------
def __Class_Practice():
    pass

# 定义类
class Person():
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

