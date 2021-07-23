import socket

class Echo_client_v1:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def run(self, word):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            print("echo client sending words: ", word)
            s.sendall(str.encode(word))
            data = s.recv(1024)
        print("Recieved: ", str(data, encoding='utf-8'))


if __name__ == '__main__':
    host = "127.0.0.1"
    port = 3200
    echo_client = Echo_client_v1(host, port)
    echo_client.run("Hello world")
