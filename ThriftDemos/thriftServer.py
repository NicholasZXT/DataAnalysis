from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from ThriftDemos.helloThrift.service import UserService
from ThriftDemos.helloThrift.domain.ttypes import User
from ThriftDemos.helloThrift.exception.ttypes import UserNotFoundException


# 这个类就是仿照 ThriftDemos.helloThrift.service.UserService 里的 Iface 写的类，相当于接口的实现类
class UserServiceHandler:
    def __init__(self):
        self.users = {}
        self.user_ids = {}

    def hello(self):
        hello_words = "Hello to UserService written by Thrift"
        print(hello_words)
        return hello_words

    def listUser(self):
        print("listUser called...")
        return list(self.users.values())

    def save(self, user: User):
        """
        Parameters:
         - user
        """
        if user.name in self.users:
            print(f"user [{user.userId}:{user.name}] exists !")
            return False
        print(f"save new user: [{user.userId}:{user.name}]")
        self.users[user.name] = user
        self.user_ids[user.userId] = user.name
        return True

    def findUsersByName(self, name):
        """
        Parameters:
         - name
        """
        if name in self.users:
            print(f"found user: {name}")
            return self.users[name]
        else:
            print(f"not found user: {name}")
            return None
        pass

    def deleteByUserId(self, userId):
        """
        Parameters:
         - userId
        """
        if userId in self.user_ids:
            userName = self.user_ids[userId]
            user = self.users[userName]
            self.user_ids.pop(userId)
            self.users.pop(userName)
            print(f"user [{userId}:{userName} was deleted.")
            return user
        else:
            raise UserNotFoundException(code='500', message=f'User [{userId}] not found')

    def userException(self):
        print("raised UserNotFoundException for test")
        raise UserNotFoundException(code='400', message='testing')


if __name__ == '__main__':
    handler = UserServiceHandler()
    processor = UserService.Processor(handler)
    transport = TSocket.TServerSocket(host='localhost', port=8765)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    # You could do one of these for a multithreaded server
    # server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    # server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)

    print('Starting the server...')
    server.serve()
    print('done.')