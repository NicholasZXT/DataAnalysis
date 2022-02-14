import sys
import os
from configparser import ConfigParser
import yaml
import json
import logging
import psutil
import getopt
import argparse

PWD = os.path.basename()

# ================== 系统资源监控 ============================
# import psutil
psutil.virtual_memory()
psutil.cpu_count()
psutil.Process(os.getpid()).memory_info()
print('当前进程的内存使用：{:.4f} GB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))


# ================== 文件路径 ================================
def __OS_Practice():
    pass


# 获取路径中的最后一个文件夹名称
os.path.basename('/path/to/Kaggle')
# 获取文件夹路径
os.path.dirname('/path/to/Kaggle')
# 获取当天py文件的路径
print("os.path.abspath(__file__): ", os.path.abspath(__file__))

import pkg_resources
# pkg_resources.get_distribution("os")
# pkg_resources.get_distribution("DateTime").version
# pkg_resources.get_distribution("sys").version


# ==================ini 配置文件解析=========================
# from configparser import ConfigParser
def __INI_Parse():
    pass


config_ini = ConfigParser()
ini_file = r"D:\Projects\DataAnalysis\PythonGrammar\config.ini"
config_ini.read(ini_file)
config_ini.sections()

config = ConfigParser()
config['DEFAULT'] = {'ServerAliveInterval': '45',
                     'Compression': 'yes',
                     'CompressionLevel': '9'}
config['bitbucket.org'] = {}
config['bitbucket.org']['User'] = 'hg'
config['topsecret.server.com'] = {}
topsecret = config['topsecret.server.com']
topsecret['Port'] = '50022'     # mutates the parser
topsecret['ForwardX11'] = 'no'  # same here
config['DEFAULT']['ForwardX11'] = 'yes'
# with open('config.ini', 'w') as configfile:
#   config.write(configfile)

config.sections()
config.get(section='bitbucket.org', option='User')
config.get(section='bitbucket.org', option='nothing', fallback='not exist')


# ===================YAML 配置文件解析=========================
# import yaml
def __YAML_Parse():
    pass


yaml_file = os.path.join(os.getcwd(), r"PythonGrammar\config.yaml")
with open(yaml_file, 'r+') as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)

config_yaml
t1 = config_yaml['section'][0]
t2 = config_yaml['section'][0].items()


# ===================== 日志记录logging ===========================
# import logging
def __Logging_Practice():
    pass


# ------- 默认日志配置 ---------
print("------------ 默认根日志器-----------------")
# 日志格式
# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
# 日期格式
# DATE_FORMAT = "%Y-%m-%d %H:%M:%S %p"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# 配置根记录器
# 默认输出到控制台
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
# 输出到文件
# logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

# 记录各种级别的日志
logging.debug("This is a debug log.")
logging.info("This is a info log.")
logging.warning("This is a warning log.")
logging.error("This is a error log.")
logging.critical("This is a critical log.")


# 手动设置
print("------------手动配置日志------------------")
# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
# 这个多设置了一个 handler 之后，由于这个 handler 也是输出到控制台的，所以同一条日志信息会被打印两次
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(fmt='%(asctime)s  %(name)s  %(levelname)s : %(message)s', datefmt=DATE_FORMAT)

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')


# ===================命令行参数解析================================
import getopt
import argparse
def __Args_Parse():
    pass


args = "-n n_value -m m_value --param1=param1_value --param2=param2_value".split(" ")
# args = "-n n_value -m m_value --param1=param1_value --param2=param2_value unknow_value".split(" ")

print("---------getopt usage-----------------------")
opts, pargs = getopt.getopt(args, "n:m:", ['param1=', 'param2='])
print("opts: ", opts)
print("pargs: ", pargs)

print("---------argparse usage-----------------------")
# 1. 创建一个参数解析器对象
parser = argparse.ArgumentParser(prog="config_parse", usage="test argparse", description="test how to use argparse")
# 2. 添加参数
parser.add_argument('-n', help="parameter -n")
parser.add_argument('-m', help="parameter -m")
parser.add_argument('--param1', help="parameter --param1")
parser.add_argument('--param2', help="parameter --param2")
# bool 参数的设置
parser.add_argument('--flag', help="bool flag parameter --flag", action="store_false")
# 3.1 解析参数，此方法只解析已定义的参数，如果有未定义的参数，会报错
args_res = parser.parse_args(args=args)
print("args_res: ", args_res)
print("args_res.n: ", args_res.n)
print("args_res.m: ", args_res.m)
print("args_res.param1: ", args_res.param1)
print("args_res.param2: ", args_res.param2)
# 将 args_res 转成 dict
args_res_dict = vars(args_res)
args_res_dict

# 3.2 解析参数，下面的方法不仅会解析已知参数，同时返回未知参数，不会报错
args = "-n n_value -m m_value --param1=param1_value --param2=param2_value unknow_value".split(" ")
args_known, args_unknown = parser.parse_known_args(args=args)
print("args_known: ", args_known)
print("args_unknown: ", args_unknown)

# 添加子命令解析类
subparser = parser.add_subparsers(help='sub-command help')
# 然后添加子命令解析器，注意，这里是解析器
# 添加一个子命令 sub_a
parser_sub_a = subparser.add_parser(name='sub_a', help='sub_a command help')
# 在子命令解析器中添加解析参数
parser_sub_a.add_argument('-a')
# 添加另一个子命令解析器
parser_sub_b = subparser.add_parser(name='sub_b', help='sub_b command help')
parser_sub_b.add_argument('-b')
# 参数
args_a = "-n n_value -m m_value --param1=param1_value --param2=param2_value sub_a -a a_value".split(" ")
args_b = "-n n_value -m m_value --param1=param1_value --param2=param2_value sub_b -b b_value".split(" ")
# 多个子命令，每次只能生效一个
args_a_res = parser.parse_args(args_a)
args_b_res = parser.parse_args(args_b)


# ========= 使用 zipfile 读取压缩文件 ===========
def __Zipfile_Practice():
    pass


import zipfile
path = r"D:\Desktop\光伏专项\冀北光伏\yc_meter_archives.zip"
file = zipfile.ZipFile(path, mode='r')
# 获取压缩文件内的各个子文件名称
file.namelist()
# 获取压缩文件内的各个子文件详细信息
file.infolist()
# 获取某个子文件的句柄，返回的是 ZipInfo 对象
subfile_handler = file.getinfo('yc_meter_archives')
# 打开某个子文件，返回的是 zipfile.ZipExtFile 对象，在该对象上可以执行read, readlines, readline 等方法读取数据
subfile = file.open(subfile_handler)
# 读取压缩文件的一行，为 bytes 类型
line_bytes = subfile.readline()
# 解码成字符串
line = line_bytes.decode()
line = json.loads(line)

# 关闭对象，实践中最好使用上下文管理器
subfile.close()
file.close()


def __Main_location():
    pass


if __name__ == "__main__":
    args = sys.argv
    # print("name: ", args[0])
    # print("args: ", args)

    # t = os.environ.get("es", "localhost:19200")

    # 测试 getopt
    # 命令: -n n_value -m m_value --param1=param1_value --param2=param2_value unknow_value
    # opts, pargs = getopt.getopt(sys.argv[1:], "n:m:", ['param1=', 'param2='])
    # print("opts: ", opts)
    # print("pargs: ", pargs)

    # 测试 argparser