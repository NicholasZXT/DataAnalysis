import socket
import selectors
import traceback
import logging
from time import sleep

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
# DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
# logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

class EchoClientV1:
    """
    第一版，单个客户端连接
    """
    def __init__(self, host, port, input_flag=False):
        self.host = host
        self.port = port
        self.__input_flag = input_flag

    @property
    def input_flag(self):
        return self.__input_flag

    @input_flag.setter
    def input_flag(self, value):
        assert isinstance(value, bool), "Expect a bool value"
        self.__input_flag = value

    def run(self, word=None):
        if self.input_flag:
            self.run_input()
        else:
            assert word is not None, "word parameter is None"
            self.run_word(word)

    def run_input(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            word = input("please input something: ")
            s.sendall(str.encode(word))
            print("Echo client sending words: ", word)
            print("Echo client is waiting data...")
            data = s.recv(1024)
            print("Recieved: ", str(data, encoding='utf-8'))
        print("Echo client stop.")

    def run_word(self, word):
        # 使用 with 语句管理上下文
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #     print("socket is connecting to : ", (self.host, self.port))
        #     # 客户端的socket是主动式，它会调用 connect 方法，此方法不是阻塞的
        #     s.connect((self.host, self.port))
        #     s.sendall(str.encode(word))
        #     print("Echo client sending words: ", word)
        #     print("Echo client is waiting data...")
        #     # 等待服务器返回消息时，是阻塞的
        #     data = s.recv(1024)
        #     print("Recieved: ", str(data, encoding='utf-8'))
        # print("Echo client stop.")
        # 手动管理 socket 的 close
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("socket is connecting to : ", (self.host, self.port))
        # 客户端的socket是主动式，它会调用 connect 方法，此方法不是阻塞的
        s.connect((self.host, self.port))
        s.sendall(str.encode(word))
        # --- 如果后面紧跟着重新发一次消息，会发现TCP连接是流式的，两次的消息在服务端同一个recv中收到
        # s.sendall(str.encode(word + " --repeat"))
        print("Echo client sending words: ", word)
        print("Echo client is waiting data...")
        # 等待服务器返回消息时，是阻塞的
        data = s.recv(1024)
        print("Received: ", str(data, encoding='utf-8'))
        s.close()
        print("Echo client stop.")


class EchoClientV2:
    """
    Echo客户端第 2 版
    提供多连接的客户端，使用 I/O多路复用 selector 的方式。

    这个客户端的实现有一个特别需要注意的地方！！！
    此客户端提供的功能是，为每个发送的 word 建立一个socket连接，并接受服务器返回的消息，之后就不再交互了。
    因此在处理 EVENT_READ 事件时，接受到服务器返回的消息后，就可以关闭 socket 的连接了，
    不需要 174 行附近的 else ，同时也不需要在 EVENT_WRITE 中执行 socket.close() 操作。

    如果在 EVENT_WRITE 中判断执行 socket.close() 操作，有个问题需要注意，客户端socket和服务端的socket的 EVENT_WRITE 总是就绪的，
    客户端 第一次 EVENT_WRITE就绪 中，发送完数据后，selector.select()会马上返回第二次的 EVENT_WRITE 就绪，此时没有数据需要发送，就会执行
    socket.close() 操作，在服务端socket看来，相当于客户端socket发送完数据就直接关闭了，这样客户端socket就 收不到 服务端返回的消息，
    并且服务端在接受消息时会抛异常。
    一个比较笨的解决办法是，执行 131 行的 sleep 操作，等待一下，让客户端socket发送完数据后，有时间接受返回的数据，然后再执行 close 操作。
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sel = selectors.DefaultSelector()

    def run(self, word_list):
        sleep_time = 0.3
        # 创建 一系列的socket
        logging.info("***************** start connection *******************")
        self.start_connection(word_list)
        # logging.info("sleeping for {} seconds.".format(sleep_time))
        # sleep(sleep_time)
        # 开始循环监控 socket 的 IO事件
        logging.info("*************** monitoring I/O events ****************")
        while True:
            logging.info("======================================================")
            logging.info("selector is waiting...")
            try:
                events = self.sel.select()
            except OSError as e:
                logging.info("selector time out, stop Echo client")
                # print(e)
                # traceback.print_exc()
                break
            logging.info("selector returns events of length: {}".format(len(events)))
            for key, mask in events:
                logging.info("------------------------------------------------------")
                logging.info("event.key.fileobj: {}".format(key.fileobj))
                logging.info("event.key.data: {}".format(key.data))
                logging.info("event.mask: {}".format(mask))
                self.sending_data(key, mask)
            """
            # 如果 一定要在 EVENT_WRITE 处理逻辑里执行关闭 socket 的操作，那么这里一定要 sleep，否则会因为程序执行太快，
            # EVENT_WRITE 中发送一次数据后，又马上在这里执行 socket.close()，导致服务端出现异常。
            """
            # logging.info("sleeping for {} seconds.".format(sleep_time))
            # sleep(sleep_time)
        # logging.info("sleeping for {} seconds.".format(sleep_time))
        # sleep(sleep_time)

    def start_connection(self, word_list):
        """
        对于 word_list 中的每个字符串，创建一个 socket 连接
        @param word_list: list of str
        @return:
        """
        word_len = len(word_list)
        server_addr = (self.host, self.port)
        for i in range(word_len):
            # 每个 socket 都有一个 id
            conn_id = i + 1
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logging.info("------------------------------------------------------")
            logging.info("starting connection '{} -- {}' to '{}'".format(conn_id, sock.fileno(), server_addr))
            # 设置为非阻塞的
            sock.setblocking(False)
            sock.connect_ex(server_addr)
            # 设置该 socket 上的监控事件
            monitor_events = selectors.EVENT_READ | selectors.EVENT_WRITE
            # 为该 socket 添加一些附属信息
            data = {"conn_id": conn_id, "socket_fileno": sock.fileno(), "contents": word_list[i]}
            logging.info("register socket '{} -- {}' to selector...".format(conn_id, sock.fileno()))
            self.sel.register(sock, monitor_events, data)

    def sending_data(self, key, mask):
        """
        当 socket 的 IO事件就绪后，执行相关的 IO 操作
        @param key:
        @param mask:
        @return:
        """
        sock = key.fileobj
        sock_info = key.data
        logging.info("processing socket: '{} -- {}'".format(sock_info['conn_id'], sock.fileno()))
        # 检查 socket 是否 读就绪
        if mask & selectors.EVENT_READ:
            logging.info("EVENT_READ is ready for '{} -- {}'".format(sock_info['conn_id'], sock.fileno()))
            # 这里的 recv() 方法一定没有阻塞
            recv_data = sock.recv(1024)
            if recv_data:
                # 接收到了data, 打印出来
                data_decode = str(recv_data, encoding='utf-8')
                logging.info("socket '{} -- {}' recieved '{}'".format(sock_info['conn_id'], sock.fileno(), data_decode))
            # else:
                """
                # 注意，这里的 else 不需要，因为此 socket 接收到服务器返回的消息后，就不再交互了 ------------------------ KEY
                # 如果这里添加了 else（且下面的 EVENT_WRITE 中没有关闭socket操作），
                # 就会导致 服务端socket 和 客户端socket 同时处于 EVENT_WRITE就绪 的 死锁状态，
                # 一直无法进入这个 else 分支，也就无法关闭 socket
                """
                # 没有数据需要发送了，关闭此 socket
                logging.info("selector unregister socket '{} -- {}'".format(sock_info['conn_id'], sock.fileno()))
                self.sel.unregister(sock)
                logging.info("socket '{} -- {}' close.".format(sock_info['conn_id'], sock.fileno()))
                sock.close()
        if sock.fileno() == -1:
            logging.info("socket has closed, skip to process EVENT_WRITE.")
            return None
        # 检查 socket 是否 写就绪 —— 这个判断有点多余，客户端socket的write通常总是就绪状态
        if mask & selectors.EVENT_WRITE:
            logging.info("EVENT_WRITE is ready for '{} -- {}'".format(sock_info['conn_id'], sock.fileno()))
            send_data = sock_info['contents']
            if len(send_data):
                # 如果此 socket 对应的 word 没有发送，则发送出去
                sock.sendall(str.encode(send_data))
                logging.info("socket '{} -- {}' send data '{}'".format(sock_info['conn_id'], sock.fileno(), send_data))
                # 发送完置空字符串
                sock_info['contents'] = ""
            else:
                # 已经发送过了，就不在发送了
                logging.info("socket '{} -- {}' has sent data.".format(sock_info['conn_id'], sock.fileno()))
                # 和服务端一样，客户端也不要在 EVENT_WRITE 里执行关闭 socket 的操作
                # self.sel.unregister(sock)
                # logging.info("socket '{} -- {}' finished.".format(sock_info['conn_id'], sock.fileno()))
                # sock.close()
                """
                # 如果一定要在这里判断执行 socket.close() 操作，那么必须要执行 129 行的 sleep 操作.
                """


if __name__ == '__main__':
    host = "127.0.0.1"
    port = 3200
    # echo_client = EchoClientV1(host, port)
    # echo_client.run("Hello world -- 1")
    # echo_client.run("Hello world -- 2")
    # echo_client.run("Hello world -- 3")
    # echo_client.input_flag = True
    # echo_client.run()
    # echo_client2 = EchoClientV1(host, port)
    # echo_client2.run("Hello world -- two")

    echo_client = EchoClientV2(host, port)
    word_list = ["Hello socket -- 1", "Hello socket -- 2", "Hello socket -- 3"]
    # word_list = ["Hello socket -- 1"]
    echo_client.run(word_list)
