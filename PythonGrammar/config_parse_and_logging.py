import os
import sys
from configparser import ConfigParser
import yaml

# ------------------ 系统资源监控 ----------------------------
import psutil
psutil.virtual_memory()
psutil.cpu_count()
psutil.Process(os.getpid()).memory_info()
print('当前进程的内存使用：{:.4f} GB'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))


# ------------------ 文件路径 --------------------------------
def os_practice():
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


# ------------------ini 配置文件解析-------------------------
def ini_parse():
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


# -------------------YAML 配置文件解析-------------------------
def yaml_parse():
    pass


yaml_file = r"D:\Projects\DataAnalysis\PythonGrammar\config.yaml"
with open(yaml_file, 'r+') as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)

config_yaml
t1 = config_yaml['section'][0]
t2 = config_yaml['section'][0].items()


# --------------------- 日志记录logging ---------------------------
def logging_practice():
    pass

import logging

# ------- 默认日志配置 ---------
print("------------ 默认根日志器-----------------")
# 日志格式
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# 日期格式
DATE_FORMAT = "%Y-%m-%d %H:%M:%S %p"
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


# -------------------命令行参数解析--------------------------------
import getopt
import argparse
def arg_parse():
    pass


args = "-n n_value -m m_value --param1=param1_value --param2=param2_value".split(" ")
# args = "-n n_value -m m_value --param1=param1_value --param2=param2_value unknow_value".split(" ")
opts, pargs = getopt.getopt(args, "n:m:", ['param1=', 'param2='])
print("opts: ", opts)
print("pargs: ", pargs)

print("---------argparse usage-----------------------")
parse = argparse.ArgumentParser(prog="config_parse", usage="test argparse", description="test how to use argparse")
parse.add_argument('-n', help="parameter -n")
parse.add_argument('-m', help="parameter -m")
parse.add_argument('--param1', help="parameter --param1")
parse.add_argument('--param2', help="parameter --param2")
# 只解析已定义的参数，如果有未定义的参数，会报错
args_res = parse.parse_args(args=args)
print("args_res: ", args_res)
print("args_res.n: ", args_res.n)
print("args_res.m: ", args_res.m)
print("args_res.param1: ", args_res.param1)
print("args_res.param2: ", args_res.param2)

# 下面的会解析已知参数，同时返回未知参数，不会报错
args = "-n n_value -m m_value --param1=param1_value --param2=param2_value unknow_value".split(" ")
args_known, args_unknown = parse.parse_known_args(args=args)
print("args_known: ", args_known)
print("args_unknown: ", args_unknown)



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