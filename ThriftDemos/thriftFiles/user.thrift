namespace java ThriftDemos.helloThrift.domain
namespace py ThriftDemos.helloThrift.domain
//namespace java ThriftDemos.helloThrift
//namespace py ThriftDemos.helloThrift

/**
 * 用户类
 */
struct  User {
  1:i32 userId,
  2:string name
}