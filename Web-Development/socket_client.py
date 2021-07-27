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
        if not isinstance(value, bool):
            raise TypeError("Expect a bool value")
        self.__input_flag = value

    def run(self, word=None):
        if self.input_flag:
            self.run_input()
        else:
            self.run_word(word)

    def run_input(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            word = input("please input something: ")
            print("Echo client sending words: ", word)
            s.sendall(str.encode(word))
            data = s.recv(1024)
            print("Recieved: ", str(data, encoding='utf-8'))
        print("Echo client stop")

    def run_word(self, word=None):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # 客户端的socket是主动式，它会调用 connect 方法
            s.connect((self.host, self.port))
            print("echo client sending words: ", word)
            s.sendall(str.encode(word))
            data = s.recv(1024)
        print("Recieved: ", str(data, encoding='utf-8'))


class EchoClientV2:
    """
    多客户端连接，使用 I/O多路复用 的方式
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sel = selectors.DefaultSelector()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.host, self.port))
        sock.listen()
        print("Echo server listening on {}:{}".format(self.host, self.port))
        sock.setblocking(False)
        self.sel.register(sock, selectors.EVENT_READ, data=None)
        pass


if __name__ == '__main__':
    host = "127.0.0.1"
    port = 3200
    echo_client = EchoClientV1(host, port)
    # echo_client.run("Hello world")
    echo_client.input_flag = True
    echo_client.run()
