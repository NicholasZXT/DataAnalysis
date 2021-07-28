import socket
import selectors
import traceback
import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

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
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print("socket is connecting to : ", (self.host, self.port))
            # 客户端的socket是主动式，它会调用 connect 方法，此方法不是阻塞的
            s.connect((self.host, self.port))
            s.sendall(str.encode(word))
            print("Echo client sending words: ", word)
            print("Echo client is waiting data...")
            # 等待服务器返回消息时，是阻塞的
            data = s.recv(1024)
            print("Recieved: ", str(data, encoding='utf-8'))
        print("Echo client stop.")


class EchoClientV2:
    """
    提供多连接的客户端，使用 I/O多路复用 的方式
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sel = selectors.DefaultSelector()

    def run(self, word_list):
        # 创建 一系列的socket
        logging.info("================= start connection ===================")
        self.start_connection(word_list)
        # 开始循环监控 socket 的 IO事件
        logging.info("================= monitoring I/O events ===================")
        while True:
            logging.info("selector is waiting...")
            try:
                events = self.sel.select(timeout=10)
            except OSError as e:
                logging.info("selector time out, stop Echo client")
                # logging.info(e)
                # traceback.logging.info_exc()
                break
            logging.info("selector returns events of length: {}".format(len(events)))
            for key, mask in events:
                logging.info("------------------------------------------------------")
                logging.info("event.key.fileobj: {}".format(key.fileobj))
                logging.info("event.key.data: {}".format(key.data))
                logging.info("event.mask: {}".format(mask))
                self.sending_data(key, mask)

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
            logging.info("starting connection '{} : {}' to '{}'".format(conn_id, sock.fileno(), server_addr))
            sock.setblocking(False)  # 设置为非阻塞的
            sock.connect_ex(server_addr)   #
            # 设置该 socket 上的监控事件
            monitor_events = selectors.EVENT_READ | selectors.EVENT_WRITE
            # 为该 socket 添加一些附属信息
            data = {"conn_id": conn_id, "socket_fileno": sock.fileno(), "contents": word_list[i]}
            logging.info("register socket '{}' to selector...".format(conn_id))
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
        logging.info("processing socket: {}".format(sock_info['conn_id']))
        # 检查 socket 是否 读就绪
        if mask & selectors.EVENT_READ:
            logging.info("EVENT_READ is read for {}".format(sock_info['client_socket']))
            # 这里的 recv() 方法一定没有阻塞
            recv_data = sock.recv(1024)
            if recv_data:
                # 接收到了data, 打印出来
                data_decode = str(recv_data, encoding='utf-8')
                logging.info("socket '{}' recieved '{}'".format(sock_info['conn_id'], data_decode))
            else:
                # 没有数据了，则关闭此 socket
                self.sel.unregister(sock)
                sock.close()
                logging.info("socket '{}' close.".format(sock_info['conn_id']))
        # 检查 socket 是否 写就绪
        if mask & selectors.EVENT_WRITE:
            logging.info("EVENT_WRITE is read for {}".format(sock_info['client_socket']))
            send_data = sock_info['contents']
            if len(send_data):
                # 如果此 socket 对应的 word 没有发送，则发送出去
                logging.info("socket '{}' send data '{}'".format(sock_info['conn_id'], send_data))
                sent_num = sock.sendall(str.encode(send_data))
                # 发送完置空字符串
                sock_info['contents'] = ""
            else:
                # 已经发送过了，就不在发送了，关闭socket
                self.sel.unregister(sock)
                sock.close()
                logging.info("socket '{}' finished.".format(sock_info['conn_id']))


if __name__ == '__main__':
    host = "127.0.0.1"
    port = 3200
    # echo_client = EchoClientV1(host, port)
    # echo_client.run("Hello world -- one")
    # echo_client.input_flag = True
    # echo_client.run()
    # echo_client2 = EchoClientV1(host, port)
    # echo_client2.run("Hello world -- two")
    echo_client = EchoClientV2(host, port)
    word_list = ["Hello socket -- 1", "Hello socket -- 2", "Hello socket -- 3"]
    echo_client.run(word_list)
