namespace java ThriftDemos.helloThrift.exception
namespace py ThriftDemos.helloThrift.exception
//namespace java ThriftDemos.helloThrift
//namespace py ThriftDemos.helloThrift


exception UserNotFoundException {
   1: string code;
   2: string message;
}