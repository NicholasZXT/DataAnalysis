namespace java ThriftDemos.helloThrift.service
namespace py ThriftDemos.helloThrift.service

include "user.thrift"
include "exception.thrift"


/*
 * 用户服务
 */
service UserService {

  //Hello
  string hello(),

  //获取用户列表
  list<user.User> listUser(),

  //保存用户
  bool save(1:user.User user),

  //根据name获取用户
  user.User findUsersByName(1:string name),

  //删除用户
  // 这里也要指定抛出的异常，否则服务端会报错
  user.User deleteByUserId(1:i32 userId) throws (1: exception.UserNotFoundException e)

  //测试异常抛出
  void userException()  throws (1: exception.UserNotFoundException e)
}