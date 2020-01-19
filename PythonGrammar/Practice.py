def FibNum(n):
    if n == 1:
        return 1
    elif n == 2:
        return 1
    else:
        return FibNum(n - 1) + FibNum(n - 2)


def Gen(num):
    n = 1
    while n <= num:
        yield FibNum(n)
        n = n + 1


print Gen(6)

G = Gen(6)
print G.next()
print G.next()
print G.next()
print G.next()
print G.next()
print G.next()
print 'next series'
for i in Gen(7):
    print i


#闭包
def sumone(*args):
	def sumtwo():
		s=0
		for i in args:
			s=s+i
		return s
	return sumtwo
#使用
f=sumone(1,2,3,4)
或者 sumone(1,2,3,4)返回值均一样为函数
f1=sumone(1,2,3,4)
f2=sumone(1,3,5,7)


#闭包
def countone():
	fs=[]
	for i in range(1,4):
		def f():    
		#这里没有使用参数
			return i*i   
			#i在函数f中并没有定义
		fs.append(f)  
		#这里返回的f也没有使用参数,如果使用f(),那么返回的是数值而非函数
	return fs

def counttwo():
	fs=[]
	for i in range(1,5):
		def g(j):
			#这里的函数g有参数
			def f(): 
			#仍然不使用参数
				return j*j 
				#直接调用外部的变量
			return f
		fs.append(g(i))  
		#这里使用函数g采用的是调用的形式
	return fs


#匿名函数
a=lambda x:x*x


#装饰器
def log(f):
	def fun(*args,**kw):
		#这里的参数不能少，因为fun最后要代替传入的函数f
		#返回，那么它要接受的参数和f应当一样，
		print 'call function '+f.__name__
		return f(*args,**kw)
		#这里返回f时使用的参数*args和**kw是在fun定义时记录
		#的参数*args和**kw，虽然这些参数在fun里面并没有用到
		#但是最后是要传给f的
	return fun

@log
def add(x,y):
	return x+y


#含参数的装饰器
def log2(text):
	#这里的text为参数，它不再是传入的函数
	def decorator(f):
		def fun (*args,**kw):
			print '%s function %s'%(text,f.__name__)
			return f(*args,**kw)
		return fun
	return decorator

@log2('execute')
#这里使用参数
def add2(x,y):
	return x+y

#完善装饰器
import funtools
def log2(text):
	#这里的text为参数，它不再是传入的函数
	def decorator(f):

		@funtools.wraps(f)
		#这里复制原函数所有属性
		def fun (*args,**kw):
			print '%s function %s'%(text,f.__name__)
			return f(*args,**kw)
		return fun
	return decorator

@log2('execute')
#这里使用参数
def add2(x,y):
	return x+y


#我的练习
def Add(f):
	def fun(*args,**kw):
		print 'The Point is '
		return f(*args,**kw)
	return fun
@Add
def Point(x,y):
	print '(%d,%d)'%(x,y)
	return None#这个返回值不能少


def log(text):
	def decorator(f):
		def fun(*args,**kw):
			print 'We add the text: %s'%text
			print 'The Point is:'
			return f(*args,**kw)
		return fun
	return decorator
@log('parameter')
def Point(x,y):
	print '(%d,%d)'%(x,y)
	return None

#全局变量的调用
x=10
def change():
	x=20
	print x

def change():
	print x
	x=20
	print x


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as plt

import numpy as np
names=np.array(['Bob','Joe','Will','Joe','Joe'])


b = np.arange(6).reshape(2,3)