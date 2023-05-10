# Thrift Python 使用示例

## 生成服务端和客户端 stub
利用 `ThriftDemos.thriftFiles` 里的 `.thrift` 文件生成python代码：
```shell
# -r 表示include的文件也包含在内
thrift.exe -r --gen py .\userService.thrift
```
生成的文件为`ThriftDemos.helloThrift`包里的所示，每个thrift文件对应生成了如下的文件：
+ `ttypes.py`：存储其中定义的`struct`对应的对象
+ `constants.py`：可能是存储常量
+ `xxxService.py`：对应的是thrift文件中`service`关键字定义的接口对象，有的话才会生成

需要注意的是，3个thrift文件中，最好不要使用同样的包命名空间，比如`namespace py ThriftDemos.helloThrift`，因为在python下，每个thrift文件都会
生成一个各自的`ttypes.py`文件，存放各自用到的`struct`对应的对象，如果都使用同样的包命名空间的话，会相互**覆盖**干扰，所以这里分别使用了3个子命名空间。

## 编写服务端和客户端代码
生成stubs文件之后，需要使用这些文件来编写客户端和服务端代码，其中最有用的是thrift文件中`service`关键字对应生成的类。
这里`userService.thrift`对应的是生成的`helloThrift.service.UserService.py`文件，其中定义的`service UserService`会生成如下3个有用的类：
+ `class Iface`：这个是一个接口类，定义了服务端的类需要实现的接口，不过由于python中没有接口这个概念，所以这个类一般是一个空壳的示例类， 
  其中填入了需要实现的方法名称，方法体为空。实现服务端业务逻辑时，只要创建一个类，在其中对应方法（方法名必须一致）里写业务逻辑即可。
+ `class Client(Iface)`：这个就是客户端的实现类，这个类在编写客户端时，直接使用即可。
+ `class Processor(Iface, TProcessor)`：这个是服务端用于封装业务处理流程的类，它实例化时接受的参数就是实现了`Iface`类中各个方法的类

不管是服务端还是客户端，在使用之前，都遵循固定的RPC连接创建流程：
1. 创建一个socket对象，位于`thrift.transport.TSocket`包里的对象，服务端为`TServerSocket`，客户端为`TSocket`
2. 创建一个transport对象，位于`thrift.transport.TTransport`包里的对象
3. 指定protocol对象，位于`thrift.protocol`包里的对象

对于服务端来说，需要按照`Iface`类的样子（实现其中的各个方法），写一个业务处理的类，然后将这个类作为参数，传递给`Processor`类，用于执行服务端的处理逻辑；

对于客户端来说，只需要使用上面的RPC连接来实例化`Client`对象，在这个对象上执行对应的方法调用即可，就像调用本地方法一样。
