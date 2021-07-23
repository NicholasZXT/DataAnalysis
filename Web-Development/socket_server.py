import socket

class Echo_server_v1:
    """
    echo server 第一版
    """
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            print("echo server is listening...")
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    print("echo server recieved: ", str(data, encoding='utf-8'))
                    conn.sendall(data)


if __name__ == '__main__':
    host = "127.0.0.1"
    port = 3200
    echo_server = Echo_server_v1(host, port)
    echo_server.run()