import socket
import selectors

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
        self.start_connection(word_list)
        while True:
            print("selector is waiting...")
            events = self.sel.select()
            for key, mask in events:
                self.sending_data(key, mask)

    def start_connection(self, word_list):
        word_len = len(word_list)
        server_addr = (self.host, self.port)
        for i in range(word_len):
            conn_id = i + 1
            print('starting connection', conn_id, 'to', server_addr)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setblocking(False)
            sock.connect_ex(server_addr)
            monitor_events = selectors.EVENT_READ | selectors.EVENT_WRITE
            data = {"conn_id": conn_id, "socket_fileno": sock.fileno(), "contents": word_list[i]}
            self.sel.register(sock, monitor_events, data)

    def sending_data(self, key, mask):
        sock = key.fileobj
        sock_info = key.data
        # 检查 socket 是否 读就绪
        if mask & selectors.EVENT_READ:
            # 这里的 recv() 方法一定没有阻塞
            recv_data = sock.recv(1024)
            if recv_data:
                # 接收到了data, 处理一下，放入 socke_info 里
                data_decode = str(recv_data, encoding='utf-8')
                print("Echo server recieved '{}' from {}".format(data_decode, sock_info['client_addr']))
                sock_info['contents'] += data_decode
            else:
                # 没收到data，说明客户端关闭了连接，此时服务端也要关闭连接，但是在此之前，要将其从 selector 中移除
                print("Echo server closing connection to :", sock_info)
                self.sel.unregister(sock)
                sock.close()
        # 检查 socket 是否 写就绪
        if mask & selectors.EVENT_WRITE:
            echo_data = sock_info['contents']
            if len(echo_data):
                print("Echo server returns '{}' to {}".format(echo_data, sock_info['client_addr']))
                # send 方法不一定能成功发送所有的数据
                sent_num = sock.send(echo_data)
                sock_info['contents'] = sock_info['contents'][sent_num:]


if __name__ == '__main__':
    host = "127.0.0.1"
    port = 3200
    echo_client = EchoClientV1(host, port)
    echo_client.run("Hello world -- one")
    # echo_client.input_flag = True
    # echo_client.run()
    # echo_client2 = EchoClientV1(host, port)
    # echo_client2.run("Hello world -- two")
