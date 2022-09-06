from urllib import parse
from sqlalchemy import create_engine, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker


# --------------- 1. 连接数据库 ---------------
def __Connection():
    pass


mysql_conf = {
    'host': 'localhost',
    'user': 'root',
    'passwd': 'mysql@2018',
    'port': 3306,
    'database': 'crashcourse'
}
# 密码里的特殊字符需要做一些转义处理
mysql_conf['passwd'] = parse.quote_plus(mysql_conf['passwd'])
# database url的格式：dialect+driver://username:password@host:port/database
# 其中的 driver 是底层的数据库驱动，注意，SQLAlchemy 本身是 不提供 数据库驱动的，需要安装对应的驱动依赖
db_url = 'mysql+pymysql://{user}:{passwd}@{host}:{port}/{database}'.format(**mysql_conf)
# 创建数据库的连接对象Engine，注意，此时并未执行连接操作
# Engine 包括数据库连接池 （Pool) 和 方言 (Dialect，指不同数据库 sql 语句等的语法差异)，两者一起以符合 DBAPI 规范的方式与数据库交互
# engine = create_engine(db_url)
# 设置 echo=True 的话，每一步则会打印出底层实际执行的SQL
engine = create_engine(db_url, echo=True)
# 获取数据库里的表名称
# engine.table_names()

# 有了`Engine`对象之后，可以通过如下两种方式执行对数据库的操作（[Working with Engines and Connections](https://docs.sqlalchemy.org/en/14/core/connections.html)）：
# 1. `Connection`对象：这个就是类似于 PEP-249 规范里定义的使用方式，通过`Connection`对象创建`Cursor`对象，执行SQL语句
# 2. `Session`对象：这个通常和SQLAlchemy-ORM配合使用


# 1.1 创建连接对象
with engine.connect() as connection:
    # .connect 返回的是 Connection 对象，在上面执行 execute 方法，传入原生的 SQL 语句
    # result = connection.execute("select * from customers")
    # 不过推荐的做法是，使用 Text 对象来封装一下 SQL 语句
    result = connection.execute(Text("select * from customers"))
    # execute() 返回的是 CursorResult 对象，可以通过迭代的方式读取其中的数据
    for row in result:
        print("row:", row)
print(row.__class__)
# sqlalchemy.engine.row.LegacyRow
print(row.items())
# [('cust_id', 10005), ('cust_name', 'E Fudd'), ('cust_address', '4545 53rd Street'), ('cust_city', 'Chicago')]
print(row.values())
# [10005, 'E Fudd', '4545 53rd Street', 'Chicago', 'IL', '54545', 'USA', 'E Fudd', None]


# 1.2 使用游标
# 游标对象不是SQLAlchemy提供的，而是由底层的数据库驱动提供的，所以要获取游标对象，需要先获取底层驱动原生的数据库连接对象
connection = engine.raw_connection()
try:
    cursor_obj = connection.cursor()
    res_num = cursor_obj.execute("select * from customers")
    print(f"res_num: {res_num}")
    results = list(cursor_obj.fetchall())
    print("results:")
    print(results)
    cursor_obj.close()
    connection.commit()
finally:
    connection.close()
# 此外，也可以通过 SQLAlchemy 的 Connection 对象的 .connection 属性获取底层的驱动的 Connection 对象，不过这个方式好像有些限制
try:
    con = engine.connect()
    print(con.__class__)
    # <class 'sqlalchemy.engine.base.Connection'>
    connection = con.connection
    print(connection.__class__)
    # <class 'sqlalchemy.pool.base._ConnectionFairy'>
    cursor_obj = connection.cursor()
    res_num = cursor_obj.execute("select * from customers")
    print(f"res_num: {res_num}")
    results = list(cursor_obj.fetchall())
    print("results:")
    print(results)
    cursor_obj.close()
    connection.commit()
finally:
    connection.close()
    con.close()


# 1.3 使用事务
# Connection.begin() 方法会返回一个事务对象 Transaction，该对象有 .close(), .commit(), .rollback() 方法，
# 但是没有.begin()方法，因为 Connection.begin() 的时候就已经表示事务的开启了
# 事务通常和上下文管理一起使用
with engine.connect() as connection:
    with connection.begin():
        connection.execute(Text("select * from customers"))
# 一个简便的写法是：
with engine.begin() as connection:
    connection.execute(Text("select * from customers"))


# 2. --------------- 建立映射 ---------------
def __Mapping():
    pass
# 建立类和数据库表的映射关系，有两种定义映射的方式：
# 1. Declarative Mapping：这个是新版的风格，即 ORM 风格 —— 推荐这个
# 2. Classical Mappings：这个是旧版的风格，更加底层，使用方式类似于原生SQL，从1.4版本开始，这个又被称为 Imperative Mappings

# 2.1 ------- 申明式定义(Declarative Mapping) ---------
# 通过 declarative_base() 函数创建 Base 类, Base 类本质上是 一个 registry 对象，它作为所有 model 类的父类，将在子类中把声明式映射过程作用于其子类
# 这个 Base 类整个程序中通常只有一个
Base = declarative_base()

# 继承 Base 类，构建映射关系
# 映射表的类被创建的时候，Base类会将定义中的所有Columne对象——也就是具体字段，改写为描述符
class User(Base):
    # 类属性 __tablename__ 定义了表名称
    __tablename__ = 'users'

    # 下面的这个属性是为了让 User 可以修改，重复定义，否则修改字段后，重新生成此类时，Base 类不允许重新注册已存在的类
    # __table_args__ = {'extend_existing': True}

    # 定义表的各个字段
    id = Column(Integer, primary_key=True)  # 主键
    name = Column(String(64))
    fullname = Column(String(64))
    nickname = Column(String(64))

    def __repr__(self):
        return "<User(name='%s', fullname='%s', nickname='%s')>" % (self.name, self.fullname, self.nickname)


# 上述定义的映射类，会生成一个 __table__ 属性，存放的是 Table 对象，记录了该表的元数据，也就是 类与表的映射关系
User.__table__
# Table 对象是 Classical Mapping 里定义映射的底层实现 ---- KEY
# 上述的 Table 对象，又属于 MetaData 这个集合的一部分，它可以通过 Base 类的.metadata 属性访问
print(Base.metadata)
# 结果为：MetaData()

# MetaData 对象实际上是一个 registry，它保存了已注册的表对应的ORM对象，同时也提供了一些用来操作表的API
# .bind 属性：底层绑定的 Engine 或者 Connection 对象
Base.metadata.bind

# .tables 属性：输出当前已注册的所有表对象
Base.metadata.tables

# .clear() 方法：清除 MetaData 中所有注册的表，注意，这个操作不会影响数据库
Base.metadata.clear()
# 上面的 User 类如果没有 __table_args__ = {'extend_existing': True} 的话，就只能生成并注册一次，除非使用上面的 clear() 方法清除
# .remove(table) 方法：清除MetaData指定的表
Base.metadata.remove(User.__table__)

# .create_all() 方法：在数据库中创建 MetaData中注册的所有表，它有一个 tables= 参数(list of Table 对象)，可以指定创建的表，
# 默认下，只会创建不存在的表，
# Base.metadata.create_all(engine)
Base.metadata.create_all(bind=engine, tables=[User.__table__])

# .drop_all() 方法：清除数据库中所有已创建的表对象，也可以接受一个 tables 参数
Base.metadata.drop_all(bind=engine)

# .reflect() 方法：从数据库中加载所有的表定义
Base.metadata.reflect(bind=engine)


# 2.2 ------- 传统定义(Classical Mapping) ---------
# 略，这个方式使用起来太繁琐，不推荐


# 3. --------------  初始化会话  --------------
def __Session():
    pass
# 使用 SQLAlchemy-ORM 和数据库沟通时，需要引入一个 Session 类，它通常由 sessionmaker() 这个工厂方法返回
# 所有 ORM 对象的载入和保存都需要通过session对象进行，
# 有两种方式创建 Session 对象
# 第一种，创建时直接配置engine
Session = sessionmaker(bind=engine)
# 第2种，先创建，后配置 engine
# Session = sessionmaker()
# Session.configure(bind=engine)

# 上述的 sessionmaker() 是一个工厂方法，它返回的 Session 是一个类，实例化这个类会得到一个绑定 Engine 对象的 Session 对象
session = Session()
# session的常见操作方法包括
# .begin() :开启事务，可以配合 with 使用
# .flush()：预提交，提交到数据库文件，还未写入数据库文件中
# .commit()：提交了一个事务
# .rollback()：回滚
# .close()：关闭事务

# .connection()：返回一个 Connection 对象
# .execute(): 执行原生SQL查询

# 与 ORM 相关的操作有：
# .add()
# .add_all()
# .query(): 执行查询，返回一个 Query 对象，这个是最重要的查询入口


# 4. --------------  CRUD 操作  --------------
def __CRUD():
    pass

# 为了方便，这里汇总一下连接的操作
mysql_conf = {
    'host': 'localhost',
    'user': 'root',
    'passwd': 'mysql@2018',
    'port': 3306,
    'database': 'crashcourse'
}
mysql_conf['passwd'] = parse.quote_plus(mysql_conf['passwd'])
db_url = 'mysql+pymysql://{user}:{passwd}@{host}:{port}/{database}'.format(**mysql_conf)
# engine = create_engine(db_url)
engine = create_engine(db_url, echo=True)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


# --------- 4.1 查询 ------------
# 以下的查询表均来自《MySQL必知必会》附带的数据表，下载地址为:https://forta.com/books/0672327120/
def __Query():
    pass

# 直接从数据库获取表定义，不需要手动创建表的ORM映射，方便一点
Base.metadata.reflect(engine)
db_tables = Base.metadata.tables
# customers 是一个 Table 对象，它是 Classical Mapping 的底层实现
customers = db_tables['customers']
# 通过 .columns （可以缩写为 .c）属性获取所有的列
list(customers.columns)
list(customers.c)
# 获取单个列
customers.c.cust_name
customers.c['cust_name']

for instance in session.query(customers).order_by().all():
    # print(instance)
    print("cust_id: ", instance.cust_id, '; cust_name: ', instance.cust_name)
print(instance.__class__)
# <class 'sqlalchemy.engine.row.Row'>

# Query 对象是使用SQLAlchemy-ORM进行查询的主要对象，它有如下常用的方法：
# .all(), .one(), first() 等返回指定数量结果
# .order_by(), .limit(), .offset()
# .count(), .distinct(), .group_by(),
# .filter(), .filter_by(), .where(), .having()
# .join(), .outerjoin()
# .union(), .union_all()
# .statement：返回Query对应的SQL语句形式
# .subguery()：返回当前Query对象对于的SQL语句，但是依旧封装为Query对象，通常用于子查询


def __Add_Update_Delete():
    pass
# --------- 4.2 插入对象 ------------
# 创建一个 User 类的对象，对应于表中的一行记录
ed_user = User(name='ed', fullname='Ed Jones', nickname='edsnickname')
ed_user.id
ed_user.name
# 插入数据库
session.add(ed_user)
session.add_all([User(name='wendy', fullname='Wendy Williams', nickname='windy'),
                 User(name='mary', fullname='Mary Contrary', nickname='mary')])
# 提交变更
session.commit()




