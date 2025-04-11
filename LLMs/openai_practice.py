"""
OpenAI的Python SDK 在 v1.0.0 版本重写过一次，使用方式发生了改变，
参见 [v1.0.0 Migration Guide #742](https://github.com/openai/openai-python/discussions/742)
具体来说，最大的变化是原来使用全局客户端的方式，现在需要手动初始化一个客户端了：
比如旧版本如下：
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
openai.base_url = "https://..."
completion = openai.Completion.create(model='curie')
新版本需要手动初始化一个客户端：
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)
client.completions.create(model='curie')

不过新版本SDK还是在整个模块的 __init__.py 文件中定义了全局客户端 _ModuleClient，并且在 _module_client.py 里定义了代理访问对象
因此新版本里，如果图省事不想创建OpenAI客户端的话，可以使用如下方式：
openai.api_key = os.environ['OPENAI_API_KEY']   # 全局客户端的配置
openai.base_url = "https://..."
from openai import chat, completions   # 这两个是全局客户端 _ModuleClient 对象的属性代理
chat.completions.create(model='curie')
completions.create(model='curie')
"""

from openai import OpenAI, AsyncOpenAI, Stream, AsyncStream, Client, AsyncClient
from openai.types import Completion
from openai.types.chat.chat_completion import ChatCompletion
# Client 和 AsyncClient 只是别名，它们分别对应于 OpenAI, AsyncOpenAI
# 查看源码可以发现，OpenAI 底层使用的是 httpx 库

# ----------- 下面的介绍以新版本的 openai SDK 为例 -------------
# api_key 不能为None，会抛异常，也不能为空字符串，否则构造 Bearer 时会抛异常；本地模型无需验证时，随便填一个字符串即可
API_KEY = 'Random'
LLM_URL = 'http://172.16.0.32:10086/v1'

client = OpenAI(
    api_key=API_KEY,
    base_url=LLM_URL,
)

# ----------------- Completion -----------------
res_cp: Completion = client.completions.create(
    model='Qwen2.5-32B-Instruct',
    prompt='Hello world',
    stream=False
)
print(res_cp)  # Completion 类型其实只是 pydantic.BaseModel 的子类
print(res_cp.id)
print(res_cp.model)
print(res_cp.usage)
print(res_cp.choices)
choice = res_cp.choices[0]
print(choice)
print(choice.text)
for choice in res_cp.choices:
    print(choice.text)
# 或者直接输出 json 字符串/dict，简单粗暴
print(choice.json())
print(res_cp.json())
print(choice.to_dict())
print(res_cp.to_dict())


# ----------------- ChatCompletion -----------------
res_chat: ChatCompletion = client.chat.completions.create(
    model='Qwen2.5-32B-Instruct',
    messages=[
        {'role': 'system', 'content': '你是一个机器学习方面的专家'},
        {'role': 'user', 'content': '请问什么是SVM算法'},
    ],
    max_tokens=2048,
    stream=False
)
print(res_chat)
print(res_chat.id)
print(res_chat.model)
print(res_chat.usage)
print(res_chat.choices)
for choice in res_chat.choices:
    # 注意，这里是 message 属性，不是 text 属性了
    # print(choice.message)
    print(choice.message.content)
print(res_chat.json())
print(res_chat.to_dict())


