from urllib import parse
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker

# --------------- 1. 连接数据库 ---------------
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
db_url = 'mysql+pymysql://{user}:{passwd}@{host}:{port}/{database}'.format(**mysql_conf)
# 创建数据库的连接对象Engine，注意，此时并未执行连接操作
# Engine 包括数据库连接池 （Pool) 和 方言 (Dialect，指不同数据库 sql 语句等的语法差异)，两者一起以符合 DBAPI 规范的方式与数据库交互
engine = create_engine(db_url)
# 设置 echo=True 的话，每一步则会打印出底层实际执行的SQL
# engine = create_engine(db_url, echo=True)

# 2. --------------- 建立映射 ---------------
# 建立类和数据库表的映射关系，有两种定义映射的方式：
# 1. Declarative Mapping：这个是新版的风格，即 ORM 风格 —— 推荐这个
# 2. Classical Mappings：这个是旧版的风格，更加底层，使用方式类似于原生SQL

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


# 上述定义的映射类，会生成一个 __table__ 属性，存放的是 Table 对象，记录的该类对于的表结构
User.__table__
# 上述的 Table 对象，又属于 MetaData 这个集合的一部分，它可以通过 Base 类的.metadata 属性访问
Base.metadata.tables
# 上面的 User 类如果没有 __table_args__ = {'extend_existing': True} 的话，就只能生成并注册一次，除非使用下面的语句
Base.metadata.clear()
# MetaData 对象实际上是一个 registry，可以用来做一些注册表等操作
# 下面会在数据库中创建 User 表（如果该表不存在的话）
Base.metadata.create_all(engine)


# 2.2 ------- 传统定义(Classical Mapping) ---------
# 略


# 3. --------------  初始化会话  --------------
# 和数据库沟通时，需要引入一个 Session 类，它通常由 sessionmaker() 这个工厂方法返回
# 所有对象的载入和保存都需要通过session对象进行，
# 有两种方式创建 Session 对象
# 第一种，创建时直接配置engine
Session = sessionmaker(bind=engine)
# 第2种，先创建，后配置 engine
# Session = sessionmaker()
# Session.configure(bind=engine)
# 上述的 sessionmaker() 是一个工厂方法，它返回的 Session 是一个类
session = Session()
# session的常见操作方法包括
# .flush()：预提交，提交到数据库文件，还未写入数据库文件中
# .commit()：提交了一个事务
# .rollback()：回滚
# .close()：关闭


# 4. --------------  CRUD 操作  --------------
# 创建一个 User 类的对象，对应于表中的一行记录
ed_user = User(name='ed', fullname='Ed Jones', nickname='edsnickname')
ed_user.id
ed_user.name