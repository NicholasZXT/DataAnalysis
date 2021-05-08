from configparser import ConfigParser
import yaml

config_ini = ConfigParser()
ini_file = r"D:\Projects\DataAnalysis\PythonGrammar\config.ini"
config_ini.read(ini_file)
config_ini.sections()

yaml_file = r"D:\Projects\DataAnalysis\PythonGrammar\config.yaml"
with open(yaml_file, 'r+') as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)

config_yaml
t1 = config_yaml['section'][0]
t2 = config_yaml['section'][0].items()