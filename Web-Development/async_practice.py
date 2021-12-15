"""
python异步编程练习
"""

import logging
import socket
from selectors import DefaultSelector, EVENT_READ, EVENT_WRITE
# from asyncio import Task, Future

selector = DefaultSelector()
stopped = False
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

# ==========================================================================

class Future:
    def __init__(self):
        self.result = None
        self._callbacks = []

    def add_done_callback(self, fn):
        self._callbacks.append(fn)

    def set_result(self, result):
        self.result = result
        for fn in self._callbacks:
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
        sock = socket.socket()
        sock.setblocking(False)
        try:
            sock.connect(('host', 80))
        except BlockingIOError:
            pass
        f = Future()

        def on_connected():
            f.set_result(None)

        selector.register(sock.fileno(), EVENT_WRITE, on_connected)
        yield f
        selector.unregister(sock.fileno())
        get = "GET {} HTTP/1.0\r\nHost: {}\r\n\r\n".format(self.url, 'host')
        sock.send(get.encode('ascii'))

        global stopped
        while True:
            f = Future()

            def on_readable():
                f.set_result(sock.recv(4096))

            selector.register(sock.fileno(), EVENT_READ, on_readable)
            chunk = yield f
            selector.unregister(sock.fileno())
            if chunk:
                self.response += chunk
            else:
                urls_todo.remove(self.url)
                if not urls_todo:
                    stopped = True
                break


class Task:
    def __init__(self, coro):
        self.coro = coro
        f = Future()
        f.set_result()
        self.step()

    def step(self, future):
        try:
            next_future = self.coro.send(future.result)
        except StopIteration:
            return
        next_future.add_done_callback(self.step)


def loop_v2():
    while not stopped:
        events = selector.select()
        for event_key, event_mask in events:
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
