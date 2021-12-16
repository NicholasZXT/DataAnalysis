"""
python异步编程练习
"""

import logging
import socket
from selectors import DefaultSelector, EVENT_READ, EVENT_WRITE
# from asyncio import Task, Future

# IO多路复用
selector = DefaultSelector()
# 控制是否停止的全局变量
stopped = False
# 待抓取的URL列表
urls_todo = ['1', '2']


class Crawler_v1:
    """
    使用回调方式的异步爬虫编程
    """
    def __init__(self, url):
        self.url = url
        self.sock = None
        self.response = b''

    def fetch(self):
        self.sock = socket.socket()
        self.sock.setblocking(False)
        try:
            self.sock.connect(('host', 80))
        except BlockingIOError:
            pass
        selector.register(self.sock, EVENT_WRITE, self.connected)

    def connected(self, key, mask):
        selector.unregister(key.fd)
        get = "GET {} HTTP/1.0\r\nHost: {}\r\n\r\n".format(self.url, 'host')
        self.sock.send(get.encode('ascii'))
        selector.register(key.fd, EVENT_READ, self.read_response)

    def read_response(self, key, mask):
        global stopped
        chunk = self.sock.recv(4096)
        if chunk:
            self.response += chunk
        else:
            selector.unregister(key.fd)
            urls_todo.remove(self.url)
            if not urls_todo:
                stopped = True


def loop_v1():
    while not stopped:
        events = selector.select()
        for event_key, event_mask in events:
            callback = event_key.data
            callback(event_key, event_mask)


# if __name__ == '__main__':
#     for url in urls_todo:
#         crawler = Crawler_v1(url)
#         crawler.fetch()
#     loop_v1()

# ================= 协程的异步编程 ==============================

class Future:
    """
    用于保存异步执行结果的类
    """
    def __init__(self):
        self.result = None
        self._callbacks = []

    def add_done_callback(self, fn):
        """
        此方法会在 self.set_result() 方法里被调用。
        用于向Future对象添加回调，但是这里添加的回调函数不是执行业务代码的，实际上，这里添加的回调方法始终是后面的Task.step()
        @param fn: 当前的Future对象本身
        @return:
        """
        self._callbacks.append(fn)

    def set_result(self, result):
        """
        给当前的Future对象设置值，同时调用所有的回调方法
        @param result:
        @return:
        """
        self.result = result
        for fn in self._callbacks:
            # 注意，这里调用回调函数的时候，传入的参数是当前的Future实例对象
            fn(self)

    def __iter__(self):
        yield self
        return self.result


class Crawler_v2:
    """
    使用协程的异步爬虫编程
    """
    def __init__(self, url):
        self.url = url
        self.response = b''

    def fetch(self):
        """此方法是一个协程"""
        sock = socket.socket()
        sock.setblocking(False)
        try:
            sock.connect(('host', 80))
        except BlockingIOError:
            pass
        # 这个Future对象用于保存socket连接建立之后的结果，由于socket建立连接时，不需要存储内容，所以下面的 on_connected() 方法里设置的结果为None
        f = Future()
        def on_connected():
            # socket连接建立后执行的回调函数，往Future对象里设置值的时候，同时调用Future对象里的回调函数
            f.set_result(None)
        # 注册 socket描述符和对应的监听事件，on_conneted()方法作为附加的data放入其中
        selector.register(sock.fileno(), EVENT_WRITE, on_connected)
        # yield Future 对象，然后此协程停在这里
        yield f
        # socket可写时从select()里返回，然后执行 on_connected() 函数，设置Future对象的值，同时调用Future里的回调函数——Task.step()方法
        # 在 Task.step() 方法里，会向协程发送当前Future对象的值（这里是None），驱动协程执行——也就是执行这里yield之后的流程，直到下面 while 里yield产出新的 Future 对象
        # 能执行到这里，说明 socket 已经是可写的状态了，从selector里注销socket
        selector.unregister(sock.fileno())
        get = "GET {} HTTP/1.0\r\nHost: {}\r\n\r\n".format(self.url, 'host')
        sock.send(get.encode('ascii'))

        global stopped
        while True:
            # 新建一个Future对象，用于保存此次yield异步执行的结果
            f = Future()
            def on_readable():
                # 伴随 socket 的 EVENT_READ 事件的回调函数，它执行时，socket已经接收到了返回的数据，将该返回数据放入Future对象里，
                # 同时会调用Future里的回调函数——Task.step()
                f.set_result(sock.recv(4096))
            # 注册本轮的socket的读事件
            selector.register(sock.fileno(), EVENT_READ, on_readable)
            # yield 返回新的Future对象，协程停在此处
            chunk = yield f
            # socket 可读时从select里返回，执行注册时关联的 on_readable() 函数，将socket接受的数据设置为Future对象的值，然后调用Future里的回调函数——Task.step()
            # 在 Task.step() 方法里，会向协程发送当前Future对象的值（也就是socket接收的数据）传递给此处的 chunk，驱动协程执行，执行后续的流程，直到下一个yield
            selector.unregister(sock.fileno())
            if chunk:
                self.response += chunk
            else:
                urls_todo.remove(self.url)
                if not urls_todo:
                    stopped = True
                break


class Task:
    """
    因为协程本身的执行需要外部事件来激活和驱动，比如上面Crawler_v2.fetch().
    Task 类就是用于激活协程并且通过事件来驱动协程的执行的。
    """
    def __init__(self, coro):
        """coro就是初始化时传入的协程"""
        # 初始化时传入的协程还未激活
        self.coro = coro
        # 下面这个Future对象的作用只是为了激活协程——通过调用.send(None)的方式
        f = Future()
        # 这里设置了None之后，下面的step方法会向协程里发送None，以此激活协程
        f.set_result(None)
        self.step(f)

    def step(self, future):
        """
        此方法用于驱动协程的执行：它一方面通过 .send() 向协程发送参数Future对象里的值，另一方面获取协程执行到下一轮yield时产出的Future对象，
        然后将此方法注册为新的Furure对象的回调函数。
        @param future:
        @return:
        """
        try:
            # 这里驱动协程，一边通过 .send() 向协程发送当前Future对象里的值；一边获取协程执行到下一轮yield时产出的Future对象
            next_future = self.coro.send(future.result)
        except StopIteration:
            return
        # 拿到yield产出的新的Future对象之后，将self.step 方法本身添加为该Future对象的回调方法
        next_future.add_done_callback(self.step)


def loop_v2():
    while not stopped:
        events = selector.select()
        for event_key, event_mask in events:
            # select 每次返回事件里，event_key里的data是 on_connected() 或者 on_readable() 方法
            callback = event_key.data
            callback()


# if __name__ == '__main__':
#     for url in urls_todo:
#         crawler = Crawler_v2(url)
#         Task(crawler.fetch())
#     loop_v2()


# =====================================================================
def connect(sock, address):
    f = Future()
    sock.setblocking(False)
    try:
        sock.connect(address)
    except BlockingIOError:
        pass

    def on_connected():
        f.set_result(None)

    selector.register(sock.fileno(), EVENT_WRITE, on_connected)
    yield from f
    selector.unregister(sock.fileno())


def read(sock):
    f = Future()

    def on_readable():
        f.set_result(sock.recv(4096))

    selector.register(sock.fileno(), EVENT_READ, on_readable)
    chunk = yield from f
    selector.unregister(sock.fileno())
    return chunk


def readall(sock):
    response = []
    chunk = yield from read(sock)
    while chunk:
        response.append(chunk)
        chunk = yield from read(sock)
    return b''.join(response)


class Crawler_v3:
    """
    代码解耦的协程异步编程
    """
    def __init__(self, url):
        self.url = url
        self.response = b''

    def fetch(self):
        global stopped
        sock = socket.socket()
        yield from connect(sock, ('host', 80))
        get = "GET {} HTTP/1.0\r\nHost: {}\r\n\r\n".format(self.url, 'host')
        sock.send(get.encode('ascii'))
        self.response = yield from readall(sock)
        urls_todo.remove(self.url)
        if not urls_todo:
            stopped = True


# if __name__ == '__main__':
#     for url in urls_todo:
#         crawler = Crawler_v3(url)
#         Task(crawler.fetch())
#     loop_v2()
