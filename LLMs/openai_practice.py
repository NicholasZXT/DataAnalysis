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

import json
import copy
from openai import OpenAI, AsyncOpenAI, Stream, AsyncStream, Client, AsyncClient
from openai.types import Completion
from openai.types.chat.chat_completion import ChatCompletion
# Client 和 AsyncClient 只是别名，它们分别对应于 OpenAI, AsyncOpenAI
# 查看源码可以发现，OpenAI 底层使用的是 httpx 库

# ----------- 下面的介绍以新版本的 openai SDK 为例 -------------
# api_key 不能为None，会抛异常，也不能为空字符串，否则构造 Bearer 时会抛异常；本地模型无需验证时，随便填一个字符串即可
# --- vLLM 部署 ---
# API_KEY = 'Empty'
# LLM_URL = 'http://172.16.0.32:10086/v1'
# MODEL = 'Qwen2.5-32B-Instruct'
# --- Ollama 本地部署 ---
API_KEY = 'Empty'
LLM_URL = 'http://localhost:11434/v1'
MODEL = 'qwen2.5:7b'
# MODEL = 'qwen3:8b'

client = OpenAI(
    api_key=API_KEY,
    base_url=LLM_URL,
)

# ----------------- Completion -----------------
def completion_usage():
    res_cp: Completion = client.completions.create(
        model=MODEL,
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
def chat_completion_usage():
    res_chat: ChatCompletion = client.chat.completions.create(
        model=MODEL,
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


# ----------------- Function/Tool Calling -----------------
def function_calling_usage():

    def long_ge_serialize(obj):
        return str(obj)

    def dragon_ball_algorithm(x: int, y: int) -> int:
        return x + y + 1

    # 上述两个函数的描述，注意，函数的参数使用 JSON Schema 描述，而不是单纯的JSON
    long_ge_serialize_desc = {
        "name": "long_ge_serialize",
        "description": "使用龙格序列化方法对Python对象进行序列化，并输出字符串",
        "parameters": {
            "type": "object",
            "properties": {
                "obj": {
                    "type": "object",
                    "description": "Python对象"
                }
            },
        }
    }
    dragon_ball_algorithm_desc = {
        "name": "dragon_ball_algorithm",
        "description": "使用龙球(DragonBall)算法计算两个数字的结果",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "第一个数字"},
                "y": {"type": "integer", "description": "第二个数字"}
            },
            "required": ["x", "y"]
        }
    }

    # OpenAI SDK 早期版本使用 functions 参数，后续推荐使用 tools 参数
    functions = [
        long_ge_serialize_desc,
        dragon_ball_algorithm_desc
    ]
    # tools 参数的每个项目使用 function 的key进行了封装
    tools = [
        {"type": "function", "function": long_ge_serialize_desc},
        {"type": "function", "function": dragon_ball_algorithm_desc}
    ]

    # ------ 第 1 轮请求 ------
    messages_r1 = [
        # {'role': 'system', 'content': '你是一个算法专家'},
        {'role': 'user', 'content': '请使用龙球(DragonBall)算法计算一下 2019 和 2022 的结果'},
    ]
    res_r1: ChatCompletion = client.chat.completions.create(
        model=MODEL,
        messages=messages_r1,
        stream=False,
        # 较新版本的OpenAI SDK中，functions 和 function_call 参数被标记为  deprecated，后续推荐使用 tools 和 tool_choice 参数
        # functions=functions,
        # function_call='auto',
        # 使用 tool 相关参数，要求 vLLM 启动服务时使用  --enable-auto-tool-choice 和 --tool-call-parser 选项
        tools=tools,
        tool_choice='auto',
    )
    # res_r1_json: str = res_r1.to_json()
    res_r1_dict = res_r1.to_dict()
    print(res_r1.choices[0].finish_reason)
    # 显示：tool_calls
    print(res_r1.choices[0].message.content)  # content 为空
    # print(res_r1.choices[0].message.tool_calls)
    print(res_r1.choices[0].message.tool_calls[0])
    # 显示：ChatCompletionMessageToolCall(...)
    # print(res_r1.choices[0].message.tool_calls[0].to_dict())
    # 显示：{'id': 'call_u6ftltdw', 'function': {'arguments': '{"x":2019,"y":2022}', 'name': 'dragon_ball_algorithm'}, 'type': 'function', 'index': 0}
    # print(res_r1.choices[0].message.tool_calls[0].id)
    # print(res_r1.choices[0].message.tool_calls[0].function)
    # print(res_r1.choices[0].message.tool_calls[0].function.name)
    # print(res_r1.choices[0].message.tool_calls[0].function.arguments)

    fun_call_id = res_r1.choices[0].message.tool_calls[0].id
    fun_call_name = res_r1.choices[0].message.tool_calls[0].function.name
    # 注意，下面的 arguments 是字符串形式的JSON
    fun_call_args = res_r1.choices[0].message.tool_calls[0].function.arguments
    fun_call_args = json.loads(fun_call_args)
    fun_call_res = eval(fun_call_name)(**fun_call_args)
    print(fun_call_res)

    # ------ 第 2 轮请求 ------
    # 将上述函数调用结果返回给模型
    messages_r2 = copy.deepcopy(messages_r1)
    # 首先将第一次调用的 message 加入到 messages 中
    messages_r2.append(res_r1.choices[0].message)
    # messages_r2.append(res_r1.choices[0].message.to_dict())
    # 然后追加函数调用结果
    messages_r2.append({
        "role": "tool",
        "content": str(fun_call_res),
        "tool_call_id": fun_call_id
    })
    res_r2: ChatCompletion = client.chat.completions.create(
        model=MODEL,
        messages=messages_r2,
        # 第二次不需要再传入 tools/tool_choice 参数了
        # tools=tools,
        # tool_choice='auto',
    )
    res_r2_dict = res_r2.choices[0].message.to_dict()
    print(res_r2.choices[0].message.content)


def main():
    completion_usage()
    chat_completion_usage()
    function_calling_usage()


if __name__ == '__main__':
    main()
