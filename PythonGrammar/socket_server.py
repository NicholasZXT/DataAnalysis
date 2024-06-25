import socket
import selectors
import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
# DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
# logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

class EchoServerV1:
    """
    echo server 第一版
    注意标示出来的 2 处阻塞的地方，它们后续可以通过 I/O 复用来解决
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # 服务端的socket是被动式，它会使用如下的 bind, listen, accept 三个方法
            s.bind((self.host, self.port))
            s.listen()
            print('Echo server is listening on: {}, {}'.format(host, port))
            # 调用此方法会阻塞，直到接收到客户端建立的连接，发送过来新的socket  ----- 第 1 个阻塞
            conn, addr = s.accept()
            print("Echo server accept connection from: ", addr)
            with conn:
                while True:
                    print("Echo server is waiting data from: ", addr)
                    # 此方法会阻塞，等待客户端 socket 发送数据  ------ 第 2 个阻塞
                    data = conn.recv(1024)
                    if not data:
                        break
                    print("Echo server received: ", str(data, encoding='utf-8'))
                    conn.sendall(data)
        print("Echo server stop.")


class EchoServerV2:
    """
    接受多连接的服务器，使用 I/O多路复用 的方式
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sel: selectors.DefaultSelector = selectors.DefaultSelector()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.host, self.port))
        sock.listen()
        logging.info('Echo server is listening on: {}, {}'.format(host, port))
        # 必须要设置为非阻塞模式
        sock.setblocking(False)
        # 将服务端的 监听socket 注册到 selector，监听 READ 事件，同时附带的 data 用于标识此 socket
        # 如果 sel 返回的是服务端 socket，那么对应 key.data 就是这里的 data
        data = {"server_socket": sock.fileno()}
        logging.info("Echo server register LISTENING socket '{}' to selector...".format(sock.fileno()))
        self.sel.register(sock, selectors.EVENT_READ, data=data)
        logging.info("===============================================================")
        while True:
            logging.info("selector is waiting...")
            # 下面的 select 方法是阻塞的，直到有一个 socket 描述符进入IO就绪状态
            # 注意，后面还会把和客户端建立的对等 socket 注册到这个 sel 里，所以它返回的不一定是 服务端的 监听socket
            events = self.sel.select()
            logging.info("selector returns events of length: {}".format(len(events)))
            # 遍历 select 方法返回的 events 列表
            for key, mask in events:
                logging.info("------------------------------------------------------")
                logging.info("event.key.fileobj: {}".format(key.fileobj))
                logging.info("event.key.data: {}".format(key.data))
                logging.info("event.mask: {}".format(mask))
                # logging.info("key: {},\nmask: {} ".format(key, mask))
                # events 返回的不一定是 服务端的 监听socket，需要做判断
                # 实际上，第一次返回的肯定是 监听socket，表示它的 accept 方法就绪
                if "server_socket" in key.data:
                    # key.data 中含有 server_socket，说明返回的是服务端的 监听socket
                    # 可以通过下面的方式，确认此时返回的就是之前 注册的 监听socket
                    # logging.info("socket id compare: {} --- {}".format(id(sock), id(key.fileobj)))
                    # 此时 服务端监听socket的 accept() 方法一定是可执行，不会被阻塞的
                    client_sock, client_addr = key.fileobj.accept()
                    # 调用 accept_socket() 方法，处理客户端
                    self.accept_socket(client_sock, client_addr)
                else:
                    # 否则就是 客户端的 对等socket，进行相应的 I/O 操作
                    self.server_echo(key, mask)
            logging.info("===============================================================")

    def accept_socket(self, client_sock, client_addr):
        """
        建立客户端连接的方法.
        这里服务器获取到和客户端的 对等socket 对象后，由于该对象中的 recv() 方法也是阻塞的，所以会把此 socket
        也注册到 selector 中，监听 READ IO事件
        @param client_sock:
        @param client_addr:
        @return:
        """
        logging.info('accepted connection from: {} with fileno {}'.format(client_addr, client_sock.fileno()))
        # 设置为非阻塞的 socket
        client_sock.setblocking(False)
        # 这里构造 客户端对等 socket 的 data，放入 selector 中
        data = {"client_socket": client_sock.fileno(), "client_addr": client_addr, "contents": ""}
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        logging.info("register CLIENT socket '{}' to selector.".format(client_sock.fileno()))
        self.sel.register(client_sock, events, data=data)

    def server_echo(self, key, mask):
        """
        处理客户端的 IO 的方法.
        @param key: SelectorKey 对象，封装了 IO就绪的 socket
        @param mask: IO就绪的事件掩码
        @return:
        """
        # 获取 SelectorKey 中的 socket 和相关的 data
        client_sock = key.fileobj
        sock_info = key.data
        logging.info("processing connection from : {}".format(sock_info['client_socket']))
        # 检查 socket 是否 读就绪
        if mask & selectors.EVENT_READ:
            logging.info("EVENT_READ is ready for {}".format(sock_info['client_socket']))
            # 这里的 recv() 方法一定没有阻塞
            recv_data = client_sock.recv(1024)
            data_decode = str(recv_data, encoding='utf-8')
            if len(data_decode):
                # 接收到了data, 处理一下，放入 socke_info 里
                logging.info("Echo server received '{}' from '{}'".format(data_decode, sock_info['client_socket']))
                sock_info['contents'] = data_decode
            else:
                # 没收到data，说明客户端关闭了连接，此时服务端也要关闭连接，但是在此之前，要将其从 selector 中移除
                # logging.info("Echo server closing connection to : {}".format(sock_info))
                self.sel.unregister(client_sock)
                logging.info("Received nothing, selector unregister and close socket '{}'.".format(client_sock.fileno()))
                client_sock.close()
        if client_sock.fileno() == -1:
            logging.info("socket has closed, skip to process EVENT_WRITE.")
            return None
        # 检查 socket 是否 写就绪 —— 这个判断有点多余，客户端socket的write通常总是就绪状态
        if mask & selectors.EVENT_WRITE:
            logging.info("EVENT_WRITE is ready for {}".format(sock_info['client_socket']))
            echo_data = sock_info['contents']
            if len(echo_data):
                # send 方法不一定能成功发送所有的数据
                # sent_num = client_sock.send(str.encode(echo_data))
                # sock_info['contents'] = sock_info['contents'][sent_num:]
                client_sock.sendall(str.encode(echo_data))
                logging.info("Echo server returns '{}' to '{}'".format(echo_data, sock_info['client_socket']))
                sock_info['contents'] = ""
            else:
                logging.info("there is no data to send")
                # 就这个服务的功能而言，服务端不要在 EVENT_WRITE 里执行关闭socket的操作


if __name__ == '__main__':
    host = "127.0.0.1"
    port = 3200
    # echo_server = EchoServerV1(host, port)
    echo_server = EchoServerV2(host, port)
    echo_server.run()
