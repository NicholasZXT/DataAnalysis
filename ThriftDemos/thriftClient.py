from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from ThriftDemos.helloThrift.service import UserService
from ThriftDemos.helloThrift.domain.ttypes import User
from ThriftDemos.helloThrift.exception.ttypes import UserNotFoundException

def main():
    # Make socket
    transport = TSocket.TSocket('localhost', 8765)
    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)
    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    # Create a client to use the protocol encoder
    client = UserService.Client(protocol)
    # Connect!
    transport.open()

    # 执行业务方法
    print("------ hello -------")
    print(client.hello())
    print("------ listUser -------")
    users = client.listUser()
    for user in users:
        print(user)
    print("------ save -------")
    u1 = User(userId=101, name='Python')
    u2 = User(userId=102, name='C')
    u3 = User(userId=103, name='C++')
    u4 = User(userId=104, name='Java')
    print(client.save(u1))
    print(client.save(u2))
    print(client.save(u3))
    print(client.save(u4))
    print(client.save(u4))
    print("------ listUser -------")
    users = client.listUser()
    for user in users:
        print(user)
    print("------ findUsersByName -------")
    user = client.findUsersByName(u1.name)
    print(user)
    print("------ deleteByUserId -------")
    user = client.deleteByUserId(u4.userId)
    print(user)
    try:
        user = client.deleteByUserId(u4.userId)
    except UserNotFoundException as e:
        print(e)
    try:
        user = client.deleteByUserId(105)
    except UserNotFoundException as e:
        print(e)
    print("------ listUser -------")
    users = client.listUser()
    for user in users:
        print(user)
    print("------ userException -------")
    try:
        client.userException()
    except UserNotFoundException as e:
        print(e)

    # Close!
    transport.close()


if __name__ == '__main__':
    main()