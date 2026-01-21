"""
LangChain 入门使用练习，适用于 v0.3.x 和 v1.x 版本.
"""
# %%
import os
import asyncio
from typing import Optional, Dict, List, Union, Any, Callable
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field
from dataclasses import dataclass
# ---------- 模型包装器抽象基类（langchain-core提供） ----------
from langchain_core.language_models.base import BaseLanguageModel  # 下面所有模型的抽象基类
from langchain_core.language_models.llms import BaseLLM, LLM  # LLM 继承自 BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel  # SimpleChatModel 继承自 BaseChatModel
# ---------- LLM 模型包装器实现类 ----------
# from langchain.llms import OpenAI, ChatGLM, Tongyi, Ollama, VLLM  # 这个用法过时了，它只是从下面的 langchain_community.llms 中导入对应对象
# from langchain_community.llms import OpenAI, Ollama
from langchain_community.llms import ChatGLM, Tongyi, VLLM
# langchain_community.llms 其实是从下面位置导入的包装器对象
# from langchain_community.llms.openai import OpenAI
# from langchain_community.llms.chatglm import ChatGLM
# from langchain_community.llms.tongyi import Tongyi
# from langchain_community.llms.ollama import Ollama
# from langchain_community.llms.vllm import VLLM
# 但是对于 **一线模型厂商**，有专门的langchain包，建议直接从对应的第三方包里导入
from langchain_openai.llms import OpenAI
from langchain_ollama.llms import OllamaLLM
# ---------- ChatLLM 模型包装器实现类 ----------
# from langchain_community.chat_models import ChatOpenAI, ChatOllama
# from langchain_community.chat_models import ChatLlamaCpp, ChatTongyi, ChatHuggingFace
# 对于 **一线模型厂商**，有专门的langchain包，建议直接从对应的第三方包里导入
from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
# ---- langchain v1.x 提供的模型统一初始化函数 ---
from langchain.chat_models import init_chat_model
# ---------- Message + Prompt 核心抽象 ----------
from langchain_core.messages import ChatMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage, FunctionMessage
from langchain_core.prompts import StringPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder, ChatMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, FewShotChatMessagePromptTemplate
# from langchain_core.prompts import PipelinePromptTemplate
# ---------- OutputParser  ----------
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser, MarkdownListOutputParser
from langchain_core.output_parsers import JsonOutputKeyToolsParser, JsonOutputToolsParser, PydanticToolsParser
# ---------- 工具调用相关组件 ----------
from langchain_core.tools import BaseTool, BaseToolkit, Tool, StructuredTool, tool, InjectedToolArg, ToolException
# langchain.tools 包里也导入了 langchian_core.tools 包里的一些内容
# from langchain.tools import BaseTool, tool, InjectedToolArg, ToolException
from langchain.tools import InjectedState, InjectedStore, ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
# community 包里提供了一些常用工具的实现
from langchain_community.tools import ListDirectoryTool, ReadFileTool, WriteFileTool, HumanInputRun, ShellTool
# ---------- 底层Runnable抽象接口 ----------
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSequence, RunnableBinding, RunnableParallel
from langchain_core.runnables.passthrough import RunnablePassthrough, RunnableAssign, RunnablePick
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager, StdOutCallbackHandler
# from langchain_core.tracers.schemas import Run
# ---------- 对话历史相关组件 ----------
# --- 以下两个组件在 v1.x 版本继续存在 ---
from langchain_core.runnables import RunnableWithMessageHistory
# chat_history 里的组件是配合早期的 memory 模块使用的，由于 memory 模块被废弃了，所以相关组件也不推荐使用了。
from langchain_core.chat_history import BaseChatMessageHistory
# 下面两个组件就是 BaseChatMessageHistory 的实现类
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
# --- 以下组件在 v1.x 版本已经不推荐使用了，并被移动到 langchain_classic 包中 ---
# from langchain.chains.llm import LLMChain
# from langchain_core.memory import BaseMemory
# from langchain.memory import ConversationBufferMemory
from langchain_classic.chains.llm import LLMChain
from langchain_classic.base_memory import BaseMemory
from langchain_classic.memory import ConversationBufferMemory
# ---------- 文档解析及加载（RAG相关） ----------
# langchain-core定义了相关的接口，具体实现大部分都交给了 langchain_community 包
# --- 文档加载&转换 ---
from langchain_core.documents import Document, BaseDocumentCompressor, BaseDocumentTransformer
from langchain_core.document_loaders import BaseLoader, BaseBlobParser, BlobLoader, Blob
from langchain_community.document_loaders import (
    TextLoader, CSVLoader, JSONLoader, WebBaseLoader, PyPDFLoader, PyMuPDFLoader
)
from langchain_community.document_transformers import (
    BeautifulSoupTransformer, Html2TextTransformer, MarkdownifyTransformer
)
# --- 文档转换（Text-Splitter），这个是由单独的 langchain-text-splitters 包提供的 ---
from langchain_text_splitters.base import TextSplitter
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, RecursiveJsonSplitter, MarkdownTextSplitter,
    NLTKTextSplitter, SpacyTextSplitter, SentenceTransformersTokenTextSplitter
)
# --- Embedding生成 ---
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain.embeddings import init_embeddings  # langchain 包里只提供了一个通用Embedding初始化函数
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
# --- 向量化存储&检索 ---
from langchain_core.vectorstores import VectorStore, InMemoryVectorStore, VectorStoreRetriever
from langchain_community.vectorstores import (
    Chroma, FAISS, Milvus, DuckDB, Redis, SKLearnVectorStore,
    ElasticsearchStore, ElasticVectorSearch, ElasticKnnSearch
)
from langchain_elasticsearch import ElasticsearchStore
# --- 文档检索 ---
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import (
    BM25Retriever, ElasticSearchBM25Retriever, KNNRetriever, MilvusRetriever, SVMRetriever
)
from langchain_elasticsearch import ElasticsearchRetriever
# ---------- 其他 ----------
# from langchain.globals import set_verbose
# from langchain.callbacks.tracers import ConsoleCallbackHandler
# ---------- v1.0 里统一的 agent 创建API ----------
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
# ---------- middleware，v1.0版本一个更新亮点 ----------
from langchain.agents.middleware import (
    AgentMiddleware, AgentState, ModelRequest, ModelResponse,
    before_agent, after_agent, before_model, after_model, wrap_model_call, wrap_tool_call, hook_config
)
# 自带的 middleware 实现
from langchain.agents.middleware import (
    SummarizationMiddleware, HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware
)
# ---------- v1.0版本Agent底层是借助的LangGraph组件 ----------
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
# ---------- v1.0版本 Auto-Agent 搭配 MCP ----------
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext, ProgressCallback, LoggingMessageCallback
from mcp.types import LoggingMessageNotificationParams
# ======================================================================================================================
# %%
# --- 阿里百炼 ---
API_KEY = ''
LLM_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL = 'qwen-max'
# --- vLLM 部署 ---
# API_KEY = 'Empty'
# LLM_URL = 'http://172.16.0.32:10086/v1'
# MODEL = 'Qwen2.5-32B'
# MODEL = 'Qwen3-32B'
# --- Ollama 本地部署 ---
# API_KEY = 'Empty'
# LLM_URL = 'http://localhost:11434'
# MODEL = 'qwen2.5:7b'
# MODEL = 'qwen3:8b'
# MODEL = 'qwen2.5:14b'
# MODEL = 'qwen3:14b'


# %%
# ======================= LLM + ChatLLM 模型包装器 使用 =======================
def llm_usage():
    print("===> llm_usage()")
    # client_llm = OpenAI(
    #     openai_api_key=API_KEY,
    #     openai_api_base=LLM_URL,
    #     model_name=MODEL,
    #     temperature=0.7,
    #     max_tokens=512,
    #     top_p=1,
    #     streaming=False,
    #     batch_size=20,
    # )
    # Ollama 的初始化参数不太一样
    # client_llm = Ollama(
    client_llm = OllamaLLM(
        base_url=LLM_URL,
        model=MODEL,
        temperature=0.7,
        top_p=1,
        keep_alive='30m'
    )

    input_str = "请解释下机器学习算法SVM的原理"
    # 同步调用，会等到全部回答生成后才会返回结果
    res = client_llm.invoke(input=input_str)
    print(res)

    # 流式输出
    for res in client_llm.stream(input=input_str):
        print(res, end='')

    # 批量调用
    inputs = ["请解释下机器学习算法SVM的原理", "请解释下机器学习算法GBDT的原理"]
    res = client_llm.batch(inputs=inputs)
    print(res[0])
    print(res[1])

    # 也可以直接使用 __call__ 方法，这是 BaseLLM/BastChatModel 提供的Callable调用，不过后续版本可能会移除此种调用方式
    res = client_llm(prompt=input_str)
    print(res)

# %%
def chat_llm_usage():
    print("===> chat_llm_usage()")
    # client_chat = ChatOpenAI(
    #     openai_api_key=API_KEY,
    #     openai_api_base=LLM_URL,
    #     model_name=MODEL,
    #     temperature=0.7,
    #     # max_tokens=512,  # ChatOpenAI 不支持此参数
    #     top_p=1,
    #     streaming=False,
    # )
    client_chat = ChatOllama(
        base_url=LLM_URL,
        model=MODEL,
        temperature=0.7,
        top_p=1,
        keep_alive='30m'
    )

    messages = [
        {'role': 'system', 'content': '你是一位机器学习方面的专家'},
        {'role': 'user', 'content': '请问什么是SVM算法'},
    ]

    # 同步调用
    res = client_chat.invoke(input=messages)
    print(type(res))   # <class 'langchain_core.messages.ai.AIMessage'>
    # print(res)
    print(res.content)
    # 还可以获取响应的元数据，包含模型信息
    print(res.response_metadata)
    print(res.response_metadata.get('model', None))
    print(res.response_metadata.get('model_name', None))
    print(res.response_metadata.get('model_provider', None))
    # 获取使用量
    print(res.usage_metadata)

    # content_blocks 是 v1.0 新增的字段
    print(type(res.content_blocks))    # <class 'list'>
    for content_block in res.content_blocks:
        print(type(content_block))    # <class 'dict'>
        print(content_block.keys())   # dict_keys(['type', 'text'])
        print(content_block)

    # 流式响应
    for res in client_chat.stream(input=messages):
        print(res.content, end='')
    # 流式响应每次返回的对象类型是 AIMessageChunk
    print(type(res))   # <class 'langchain_core.messages.ai.AIMessageChunk'>

    # Callable调用，不过只支持 List[BaseMessage] 参数，并且被标记为deprecated
    msg = [SystemMessage(content='你是一个机器学习方面的专家'), HumanMessage(content='请问什么是SVM算法')]
    res = client_chat(messages=msg)
    print(res.content)

# %%
def get_client_llm() -> Union[BaseLLM, LLM]:
    # client_llm = OpenAI(
    #     openai_api_key=API_KEY,
    #     openai_api_base=LLM_URL,
    #     model_name=MODEL,
    #     max_tokens=512
    # )
    # client_llm = Ollama(
    client_llm = OllamaLLM(
        base_url=LLM_URL,
        model=MODEL,
        keep_alive='30m',
        think=False
    )
    print(f"\n===> Using model '{MODEL}' with {client_llm.get_name()}\n")
    return client_llm

# %%
def get_client_chat() -> Union[BaseChatModel, SimpleChatModel]:
    client_chat = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=LLM_URL,
        model_name=MODEL,
        # temperature=0.7,
        # top_p=1,
        # streaming=False,
    )
    # client_chat = ChatOllama(
    #     base_url=LLM_URL,
    #     model=MODEL,
    #     # temperature=0.7,
    #     # top_p=1,
    #     keep_alive='30m',
    #     # 控制模型是否进行think，只对支持think的模型有效，但是似乎对 qwen3 的think模式无效
    #     think=False
    # )
    print(f"\n===> Using model '{MODEL}' with {client_chat.get_name()}\n")
    return client_chat

# %%
def simple_chat():
    print("===> simple_chat()")
    client_chat = get_client_chat()
    # print(type(client_chat))
    # issubclass(type(client_chat), BaseChatModel)
    # print(hasattr(client_chat, 'profile'))
    msg = [
        HumanMessage(content='RTX 4060 Ti 16GB跑本地大模型怎么样？'),
        # 只有手动在消息前面加上 /no_think，对于 qwen3 的think模式才会有效，但是此时仍然会输出一个空的 <think></think> 块
        # HumanMessage(content='/no_think RTX 4060 Ti 16GB跑本地大模型怎么样？'),
    ]
    # res = client_chat.invoke(input=msg)
    # print(res)
    for chunk in client_chat.stream(input=msg):
        print(chunk.content, end='')


# %%
# ======================= Message + PromptTemplate 使用 =======================
def message_usage():
    """
    对于 ChatModel 来说，每次对话的最小单元就是 Message。
    官方文档[Messages](https://python.langchain.com/docs/concepts/messages/)
    展示各类 Message 封装类的使用:
    - ChatMessage: 通用消息 —— 似乎用的不多
    - SystemMessage: 设置模型身份的消息
    - HumanMessage: 用户输入的消息
    - AIMessage: 模型返回的消息，其中可能包含 ToolCall 信息
    - ToolMessage: 工具调用返回的消息封装，会返回给模型 —— 相当一个 HumanMessage
    """
    print("===> message_usage()")
    # -------- ChatMessage 使用 --------
    # 不过 ChatMessage 的使用好像不多
    chat_msg = ChatMessage(role='user', content='Hello ChatGPT')
    print(chat_msg)
    # content='Hello ChatGPT' additional_kwargs={} response_metadata={} role='user'
    print(chat_msg.type)
    print(chat_msg.role)
    print(chat_msg.content)
    print(chat_msg.json())
    # {"content":"Hello ChatGPT","additional_kwargs":{},"response_metadata":{},"type":"chat","name":null,"id":null,"role":"user"}
    print(chat_msg.pretty_repr())
    chat_msg.pretty_print()

    # -------- SystemMessage/HumanMessage 使用 --------
    # 这两个 Message 类除了 type 属性的值不一样，其他几乎都一样
    sys_msg = SystemMessage(content='You are a helpful assistant.')
    print(sys_msg)
    # content='You are a helpful assistant.' additional_kwargs={} response_metadata={}
    print(sys_msg.type)  # system
    print(sys_msg.content)
    # print(sys_msg.role)  # 它没有 role 属性
    print(sys_msg.json())
    # {"content":"You are a helpful assistant.","additional_kwargs":{},"response_metadata":{},"type":"system","name":null,"id":null}

    print(sys_msg.pretty_repr())
    sys_msg.pretty_print()

    # -------- AIMessage 使用 --------
    # AIMessage 类有几个独有的属性：
    # - usage_metadata: 模型使用信息，是一个 TypedDict，包含：input_tokens, output_tokens, total_tokens, input_token_details 等
    # - tool_calls: list[ToolCall] : 如果触发了工具调用，则该属性有值
    #   - 每个 ToolCall 对象包含：id: str, name: str, args: dict
    # - invalid_tool_calls: 不合法的工具调用信息

    # -------- ToolMessage 使用 --------
    # ToolMessage 类用于封装工具调用的返回结果，后续会被传递给模型 —— 相当于一个 HumanMessage
    # ToolMessage 封装的工具调用结果存放在 content 属性中 —— 该属性继承自 BaseMessage
    # 此外，ToolMessage 类有几个独有的属性：
    # - tool_call_id: 工具调用的 id，需要和 AIMessage 中对应的工具调用的 id 一致
    # - status: 工具调用结果，是一个字符串：success 或者 error
    # - artifact: 工具调用的其他信息，此字段的内容不应当发送给模型
    # - response_metadata: 从 BaseMessage 继承，但是目前没有用到。

    # -------- 在Message中加入额外信息 --------
    # 使用关键字参数传入额外的信息
    msg_add = ChatMessage(role='user', content='Hello ChatGPT', thinking=True, additional_kwargs={'some': 'something'})
    msg_add = SystemMessage(content='You are a helpful assistant.', thinking=True, additional_kwargs={'some': 'something'})
    print(msg_add)

    # -------- Message ContentBlock 使用 --------
    # Langchain v1.x 新增的功能，BaseMessage 基类提供了一个名为 content_blocks 的 property，
    # 对 content 字段进行解析，返回一个标准的、类型安全的 内容表示。


# %%
def prompt_template_usage():
    """
    Completion模型使用的 PromptTemplate。
    """
    print("===> prompt_template_usage()")
    # StringPromptTemplate含有抽象方法，不能实例化
    # pt = StringPromptTemplate(input_variables=["p1", "p2"], template="content-1: {p1}, content-2: {p2}")

    # PromptTemplate 是 Completion 模型使用的基础模版
    template = "Tell me a {adjective} joke about {content}."

    # 第1种：直接实例化
    pt1 = PromptTemplate(input_variables=["adjective", "content"], template=template)
    # 一般使用如下两个方法：
    # 1. format方法直接返回字符串
    pt1.format(adjective="funny", content="chickens")
    # 2. format_prompt方法返回 <class 'langchain_core.prompt_values.StringPromptValue'>
    pv1 = pt1.format_prompt(adjective="funny", content="chickens")
    print(type(pv1))
    print(pv1)  # text='Tell me a funny joke about chickens.'
    # StringPromptValue 只需要关注如下两个方法：
    print(pv1.to_string())
    print(pv1.to_messages())

    # 第2种：使用类方法 from_template —— 推荐这种方式
    pt2 = PromptTemplate.from_template(template=template)
    pt2.format(adjective="nice", content="dog")
    print(pt2.template)
    print(pt2.template_format)
    print(pt2.input_variables)

# %%
def chat_prompt_template_usage():
    """
    聊天模型（ChatModel）使用的 PromptTemplate 模版主要有如下几个：
    - 单条消息（抽象类 BaseStringMessagePromptTemplate 的子类）：
      - ChatMessagePromptTemplate，通用消息模版，下面3个是专用的
      - HumanMessagePromptTemplate
      - AIMessagePromptTemplate
      - SystemMessagePromptTemplate
    - 多条消息，使用 ChatPromptTemplate 对上面的单条消息进行 List 封装
    """
    print("===> chat_prompt_template_usage()")
    # ----- ChatMessagePromptTemplate 使用 -----
    template1 = "Tell me a {adjective} joke about {content}."
    # ChatMessagePromptTemplate 必须要指定 template 和 role
    cmpt = ChatMessagePromptTemplate.from_template(template=template1, role="user")

    # 1. format 方法返回的是 ChatMessage 对象
    cmpt_msg = cmpt.format(adjective="nice", content="fish")
    print(type(cmpt_msg))  # <class 'langchain_core.messages.chat.ChatMessage'>
    print(cmpt_msg)  # content='Tell me a nice joke about fish.' additional_kwargs={} response_metadata={} role='user'
    print(cmpt_msg.content)
    print(cmpt_msg.type)  # chat
    print(cmpt_msg.role)  # user

    # 2. format_messages 方法，返回的是 List[ChatMessage]
    cmpt_msgs = cmpt.format_messages(adjective="nice", content="cat")
    print(type(cmpt_msgs))     # <class 'list'>
    print(type(cmpt_msgs[0]))  # <class 'langchain_core.messages.chat.ChatMessage'>
    print(cmpt_msgs)

    # 3. 其他有用方法
    print(cmpt_msg.json())
    print(cmpt_msg.pretty_repr())
    cmpt_msg.pretty_print()

    # ----- HumanMessagePromptTemplate/AIMessagePromptTemplate/SystemMessagePromptTemplate 使用 -----
    template2 = "Tell me a {desc} joke about {something}."
    hmpt = HumanMessagePromptTemplate.from_template(template=template2)
    hmpt_msg = hmpt.format(desc="good", something="dog")
    print(type(hmpt_msg))
    # <class 'langchain_core.messages.human.HumanMessage'>
    print(hmpt_msg)
    # content='Tell me a good joke about dog.' additional_kwargs={} response_metadata={}
    print(hmpt_msg.content)
    print(hmpt_msg.type)   # human
    # HumanMessage 没有 role 属性！
    # print(hmpt_msg.role)

    # --- ChatPromptTemplate 用于组合多条消息的 PromptTemplate ---
    # 使用 __init__ 方法 或者 from_messages() 方法实例化对象，实际上 from_messages() 方法底层就是直接调用的 __init__() 方法，
    # 接收一个 List，其中的元素可以是：Union[BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate]
    # 使用 List[BaseMessagePromptTemplate]/List[BaseChatPromptTemplate] 创建时，后续的 format方法会起作用
    # print(type(cmpt), type(hmpt))
    cpt = ChatPromptTemplate.from_messages(messages=[cmpt, hmpt])
    # 使用 List[BaseMessage] 创建时，后续的 format方法就没啥用了
    cpt = ChatPromptTemplate.from_messages(messages=[cmpt_msg, hmpt_msg])
    print(cpt.messages)
    print(cpt.pretty_repr())
    # cpt.pretty_repr()

    # 主要有 3 个方法：format_message, format_prompt, format
    # --- format_messages 方法，返回 list[BaseMessage] ---
    cpt_r1 = cpt.format_messages(adjective="fantastic", content="cat", desc="laugh", something="rabbit")
    for msg in cpt_r1:
        print(msg.pretty_repr())
        print(msg)
    # --- format 方法，返回 str ---
    cpt_r2 = cpt.format(adjective="fantastic", content="cat", desc="laugh", something="rabbit")
    print(cpt_r2)
    # user: Tell me a nice joke about fish.
    # Human: Tell me a good joke about dog.
    # --- format_prompt 方法，返回 PromptValue ---
    cpt_r3 = cpt.format_prompt(adjective="fantastic", content="cat", desc="laugh", something="rabbit")
    print(type(cpt_r3))
    # <class 'langchain_core.prompt_values.ChatPromptValue'>
    print(cpt_r3)
    for msg in cpt_r3.messages:
        print(msg.pretty_repr())
        print(msg)

# %%
def message_placeholder_usage():
    """
    MessagesPlaceholder，顾名思义，一个用于在提示模板（PromptTemplate）中动态插入对话历史消息的占位符组件
    注意，MessagesPlaceholder 只能用于 ChatPromptTemplate 中，不能搭配 PromptTemplate 使用。
    主要作用就是在 ChatPromptTemplate 中预留一个位置，用于在运行时动态传入一系列 BaseMessage（如 HumanMessage, AIMessage, SystemMessage 等）。
    """
    print("===> message_placeholder_usage()")
    # ------ MessagesPlaceholder 使用 ------
    # 使用 optional=True，表示这个变量是可选的，如果不传，则不会报错，但会返回空列表
    prompt = MessagesPlaceholder(variable_name="history", optional=True)
    # 如果没有 optional=True，下面会抛异常
    print(prompt.format_messages())
    # 传入的对象必须是 List[BaseMessage]
    history = [("system", "You are an AI assistant."), HumanMessage(content="Hello!")]
    res = prompt.format_messages(history=history)
    print(type(res))     # <class 'list'>
    print(type(res[0]))  # <class 'langchain_core.messages.system.SystemMessage'>
    print(type(res[1]))  # <class 'langchain_core.messages.human.HumanMessage'>
    print(res)
    for msg in res:
        print(msg.content)

    # 组合 MessagesPlaceholder + ChatPromptTemplate 使用，构造对话历史模版
    chat_prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", "你是一个智能助手，负责回答用户的问题。"),
            # 这里表示后续会通过 history 插入一系列的对话历史，history 必须是一个 List[BaseMessage]
            MessagesPlaceholder("history"),
            ("human", "{user_input}")
        ]
    )
    # 准备对话历史
    conversation_history = [
        HumanMessage(content="你好！"),
        AIMessage(content="你好！有什么我可以帮忙的吗？"),
        HumanMessage(content="今天的天气怎么样？"),
        AIMessage(content="今天天气晴朗，温度适中。"),
    ]
    # 用户当前输入
    user_input = "明天会下雨吗？"
    # --- invoke 方法 ---
    r1 = chat_prompt.invoke(input={"history": conversation_history, "user_input": user_input})
    print(type(r1))  # <class 'langchain_core.prompt_values.ChatPromptValue'>
    print(r1)
    r1_msgs = r1.to_messages()
    print(type(r1_msgs[0]))  # <class 'langchain_core.messages.system.SystemMessage'>
    for msg in r1_msgs:
        # print(msg)
        print(msg.content)
    # --- format_prompt 方法，返回值和 invoke 方法一样 ---
    formatted_prompt = chat_prompt.format_prompt(history=conversation_history, user_input=user_input)
    print(type(formatted_prompt))  # <class 'langchain_core.prompt_values.ChatPromptValue'>
    for msg in formatted_prompt.to_messages():
        # print(msg)
        print(msg.content)
    # --- format_messages 方法 ---
    formatted_msgs = chat_prompt.format_messages(history=conversation_history, user_input=user_input)
    print(type(formatted_msgs[0]))  # <class 'langchain_core.messages.system.SystemMessage'>
    for msg in formatted_msgs:
        # print(msg)
        print(msg.content)

# %%
def fewshot_prompt_template_usage():
    print("===> fewshot_prompt_template_usage()")
    # ----- FewShotPromptTemplate 使用 -----
    # 构造一个反义词接龙游戏的 FewShot 提示
    examples = [
        {'input': '快乐', 'output': '悲伤'},
        {'input': '高', 'output': '矮'},
        {'input': '胖', 'output': '瘦'},
        {'input': '黑', 'output': '白'},
    ]
    example_prompt = PromptTemplate(
        input_variables=['input', 'output'],
        template='词语：{input}\n反义词: {output}\n',
    )
    fewshot_prompt = FewShotPromptTemplate(
        # 示例
        examples=examples,
        # 示例的模板
        example_prompt=example_prompt,
        # 每个示例的分隔符
        example_separator='\n',
        # FewShot描述前缀
        prefix="请输入一个词，输出一个与之含义相反的词，以下是一些例子：\n",
        # FewShot描述的后缀
        suffix="现在轮到你了:\n词语：{input}，反义词是:",
        # 输入变量
        input_variables=['input']
    )
    fsp_r1 = fewshot_prompt.format(input='好')
    print(fsp_r1)
    fsp_r2 = fewshot_prompt.format_prompt(input='好')
    print(type(fsp_r2))
    # <class 'langchain_core.prompt_values.StringPromptValue'>
    print(fsp_r2.text)

    # ----- FewShotChatMessagePromptTemplate 使用 -----
    # 只需要将 example_prompt 的类型由 PromptTemplate 改为 ChatMessagePromptTemplate 即可
    human_template = "词语：{input}"
    ai_template = "反义词: {output}"
    human_prompt = ChatMessagePromptTemplate.from_template(template=human_template, role="user")
    ai_prompt = ChatMessagePromptTemplate.from_template(template=ai_template, role="ai")
    example_prompt_chat = ChatPromptTemplate.from_messages(messages=[human_prompt, ai_prompt])
    fewshot_prompt_chat = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt_chat,
        input_variables=['input']
        # 这个没有 prefix 和 suffix 参数了
    )
    fscp_r1 = fewshot_prompt_chat.format(input='好')
    print(fscp_r1)
    fscp_r2 = fewshot_prompt_chat.format_prompt(input='好')
    # print(fscp_r2)
    for msg in fscp_r2.messages:
        print(msg.content)
    fscp_r3 = fewshot_prompt_chat.format_messages(input='好')
    # print(fscp_r3)
    for msg in fscp_r3:
        print(msg.content)

# %%
def pipeline_prompt_usage():
    """
    PipelinePromptTemplate 被标识为 Deprecated 了，所以不做研究
    """
    ...

# ======================= Output Parser + Structured Output =======================
# %%
def output_parser_usage():
    """
    LangChain的输出解析器是和提示词配合使用的，它会在提示词的末尾增加一段要求大模型输出指定格式的指令。
    常用的有如下几种Parser:
    - StrOutputParser: 原样返回
    - JsonOutputParser: 以JSON格式返回
    - PydanticOutputParser: 以Pydantic对象返回，它继承自JsonOutputParser
    - MarkdownListOutputParser: 以MarkDown列表形式返回

    每个Parser类需要实现两个方法：
    - get_format_instructions() 方法，用于返回一段提示词，告诉模型需要返回的格式
    - parse(text: str) 方法，用于将模型的输出（字符串）解析成指定格式

    参考 v1.x 版本的OutputParser接口说明：https://reference.langchain.com/python/langchain_core/output_parsers/
    OutputParser 是早期用于解决获取模型输出格式的方案，随着大模型的发展，大部分模型都原生支持了 Structured Output 功能，
    OutputParser 的使用似乎没有那么必要了，但是Langchain选择保留下来，是为了兼容那些暂时不支持 Structured Output 的模型，
    以及提供更加深入自定义解析的功能。
    """
    print("===> output_parser_usage()")
    class MyModel(BaseModel):
        name: str
        age: int
        position: str
        achievements: List[str]

    # 先实例化一个 parser 对象，可以通过 get_format_instructions 查看该Parser的格式化提示词指令
    # parser = StrOutputParser() # 注意，StrOutputParser没有提示词，因为它原样输出
    # parser = JsonOutputParser(pydantic_object=MyModel)
    parser = PydanticOutputParser(pydantic_object=MyModel)
    # 获取格式化提示词指令
    format_instructions = parser.get_format_instructions()
    print(format_instructions)
    # 注意模版最后的 {format_instructions}，它并不在 input_variables 中填充

    template = "请简单介绍下{person}的履历，需要包含姓名，年龄，职位，成就等信息\n{format_instructions}"
    prompt = PromptTemplate(
        input_variables=['person'],
        template=template,
        # 设置 output_parser 参数，将 parser 对象传入到模板中
        output_parser=parser,
        # 通过 partial_variables 参数，将 parser 对象传入到模板中
        partial_variables={'format_instructions': format_instructions},
    )
    r1 = prompt.format(person='雷军')
    print(r1)
    r2 = prompt.format_prompt(person='雷军')
    print(r2.text)
    print(prompt.partial_variables)

    client_llm = get_client_llm()
    res = client_llm.invoke(input=prompt.format_prompt(person='雷军'))
    print(res)

    # 调用模型之后，使用如下方式解析模型输出
    res_parse = parser.parse(text=res)
    print(type(res_parse))
    print(res_parse)

    # --- 另一个例子 ----
    parser = MarkdownListOutputParser()
    format_instructions = parser.get_format_instructions()
    template = "请简单介绍下机器学习领域{ml}算法的步骤.\n{format_instructions}"
    prompt = PromptTemplate(
        input_variables=['ml'],
        template=template,
        output_parser=parser,
        partial_variables={'format_instructions': format_instructions},
    )
    res = client_llm.invoke(input=prompt.format_prompt(ml='SVM'))
    print(res)
    res_parse = parser.parse(text=res)
    print(type(res_parse))
    for item in res_parse:
        print(item)

# %%
def structured_output_usage():
    """
    展示如何输出结构化的内容，参考官方文档：
    [How-to-Guides -> How to return structured data from a model](https://python.langchain.com/docs/how_to/structured_output/)

    注意：Structured Output 和上面的 Output Parser 原理并不一样，它不是通过在提示词中告诉模型输出什么格式，
    而是借助于大模型原生的能力，在接口调用中通过 response_format (OpenAI API)、format (Ollama) 之类的
    专用参数指定大模型需要返回JSON结构的回答，因此需要对应的模型原生支持此功能。

    LangChain主要是通过 BaseLanguageModel 定义的抽象方法 with_structured_output（无默认实现）实现此功能。
    该方法接受一个 schema，用于描述模型输出的字段，返回一个 Runnable 对象。
    一般有 3 种指定 schema 的方式：
    1. TypedDict
    2. JSON Schema
    3. Pydantic Model
    使用 1和2 返回的是一个 dict，使用 3 返回的是对应 Pydantic Model 的对象

    with_structured_output() 方法的实现交给了具体的模型类，但不是所有的模型类都实现了此方法。
    特别是 LLM 类的模型，大都没有实现此方法
    **一般只有 ChatModel 有此方法，因为 BaseChatModel 提供了一个默认实现**。
    此外，查看源码可以发现，with_structured_output() 需要该模型实现 bind_tools() 方法。

    具体有哪些模型实现了，可以参考 https://python.langchain.com/docs/integrations/chat/#featured-providers 表格。
    """
    print("===> structured_output_usage()")
    class JokeDict(TypedDict):
        """Joke to tell user."""
        setup: Annotated[str, ..., "The setup of the joke"]
        # Alternatively, we could have specified setup as:
        # setup: str                    # no default, no description
        # setup: Annotated[str, ...]    # no default, no description
        # setup: Annotated[str, "foo"]  # default, no description
        punchline: Annotated[str, ..., "The punchline of the joke"]
        rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

    class JokeModel(BaseModel):
        """Joke to tell user."""
        setup: str = Field(description="The setup of the joke")
        punchline: str = Field(description="The punchline to the joke")
        rating: Optional[int] = Field(default=None, description="How funny the joke is, from 1 to 10")

    client_chat = get_client_chat()
    print(type(client_chat))
    # 检查Chat模型类是否实现了 bind_tools 方法
    print(getattr(client_chat, 'bind_tools'))
    # 检查Chat模型类是否实现了 with_structured_output 方法
    print(getattr(client_chat, 'with_structured_output'))
    # 如果使用 ChatOllama，不要使用从 langchain_community.chat_models 导入的 ChatOllama，
    # 而是使用 langchain_ollama.chat_models 里的 ChatOllama，因为前者没有实现自己的 bind_tools 方法，会报错

    structured_chat_dict = client_chat.with_structured_output(schema=JokeDict)
    # structured_chat_dict = client_chat.with_structured_output(schema=JokeDict, include_raw=True)
    res_dict = structured_chat_dict.invoke("Tell me a joke about cats")
    print(type(res_dict))
    print(res_dict)

    structured_chat_model = client_chat.with_structured_output(schema=JokeModel)
    print(type(structured_chat_model))  # <class 'langchain_core.runnables.base.RunnableSequence'>
    res_model = structured_chat_model.invoke("Tell me a joke about cats")
    print(type(res_model))  # <class '__main__.JokeModel'>
    print(res_model)

    # 如果使用了 include_raw=True，那么返回的 res_model 是一个 dict，而不是一个 Pydantic Model
    structured_chat_model = client_chat.with_structured_output(schema=JokeModel, include_raw=True)
    res_model = structured_chat_model.invoke("Tell me a joke about cats")
    print(type(res_model))   # <class 'dict'>
    print(res_model.keys())  # dict_keys(['raw', 'parsed', 'parsing_error'])
    print(type(res_model['raw']))      # <class 'langchain_core.messages.ai.AIMessage'>
    print(type(res_model['parsed']))   # <class '__main__.JokeModel'>
    print(res_model['parsing_error'])  # 解析有异常时才有值
    print(res_model['parsed'])


# ======================= 工具调用 相关模块使用 =======================
# %%
def tool_wrapper_usage():
    """
    LangChain tool使用，参考官方文档:
    v1.x 版本见：https://docs.langchain.com/oss/python/langchain/tools
    v0.3.x版本的如下：
    - [Conceptual Guide -> Tools](https://python.langchain.com/docs/concepts/tools/)
    - [Conceptual Guide -> Tool calling](https://python.langchain.com/docs/concepts/tool_calling/)
    - [How-to guides -> How to use chat models to call tools](https://python.langchain.com/docs/how_to/tool_calling/)
    - [How-to guides -> How to pass tool outputs to chat models](https://python.langchain.com/docs/how_to/tool_results_pass_to_model/)

    langchian.tools 模块里，定义了如下对 Tool 的封装类
    - BaseTool 抽象类定义了langchain里Tool必须要实现的接口，其中只有一个抽象方法 `_run()` 需要子类实现。
    - Tool：封装的工具function接收的是 **单个** 字符串格式的参数。
    - StructuredTool: 封装的工具function接收指定格式的 多个 参数，通常由一个Pydantic Model描述 —— 推荐使用这个。

    一般使用 langchain.tools.convert.py 提供的 @tool 装饰器来将一个函数封装为 BaseTool 的子类（Tool、StructuredTool），
    - 如果 infer_schema=true (默认值) 或者提供了 args_schema 参数，那么就使用 StructuredTool 来封装function；
    - 不满足上面的条件，则使用 Tool 类来封装function，此时会认为该function是一个 “a simple string->string function”
    """
    print("===> tool_wrapper_usage()")
    # ----------- Tool 使用 -----------
    # 正常版本接收 2 个 int 参数的加法函数
    def add_v1(x: int, y: int) -> int:
        """Add two numbers"""
        return x + y

    # 专门适配 Tool 类的函数：函数接收一个字符串参数，格式为 "x+y"
    def add_v2(x_y: str) -> int:
        """Add two numbers"""
        x, y = map(int, x_y.split("+"))
        return x + y

    add_tool_v1 = Tool(
        name="add",
        func=add_v1,
        description="Add two numbers",
        # args_schema=None,  # 默认就是 None
    )
    # 调用会报错：langchain_core.tools.base.ToolException: Too many arguments to single-input tool add.
    r1 = add_tool_v1.invoke(input={"x": 1, "y": 2})

    add_tool_v2 = Tool(
        name="add",
        func=add_v1,
        description="Add two numbers",
        # 手动设置 args_schema 也不好使
        args_schema={"x": int, "y": int}
    )
    # 调用还是会报错：langchain_core.tools.base.ToolException: Too many arguments to single-input tool add.
    r2 = add_tool_v2.invoke(input={"x": 1, "y": 2})

    add_tool_v3 = Tool(
        name="add",
        func=add_v2,
        description="Add two numbers",
    )
    # 调用正常
    r3 = add_tool_v3.invoke(input="1+2")
    print(r3)

    # ----------- StructuredTool 使用 -----------
    # StructuredTool 提供了一个类方法 from_function()，不需要使用初始化方法
    add_tool_struct = StructuredTool.from_function(
        func=add_v1,
        name="add",
        description="Add two numbers",
        # 可以指定自动推断参数schema
        infer_schema=True,  # 默认值
        # 也可以手动指定
        # args_schema={"x": int, "y": int}
    )
    # 调用正常
    r4 = add_tool_struct.invoke(input={"x": 1, "y": 2})
    print(r4)


# %%
def tool_usage():
    """
    一般使用 langchain.tools.convert.py 提供的 @tool 装饰器来将一个函数封装为 BaseTool 的子类（Tool、StructuredTool）
    定义工具时，和OpenAI的function calling类似，至少需要 name, description, schema 3个描述字段。
    """
    print("===> tool_usage()")
    # 使用 @tool 装饰器定义工具
    @tool(
        description="使用龙球(DragonBall)算法计算两个数字的结果",
        # args_schema 用于设置被调用函数的参数schema，有两种形式：
        # 1. Pydantic Model
        # 2. JSON Schema，用dict描述，注意，不是任意dict的形式，否则下面的 .args 属性会报错
        # 或者不设置，默认（infer_schema）会从函数参数中自动提取，此时建议参数使用 Annotated 进行注解
        args_schema={
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "第一个数字"},
                "y": {"type": "integer", "description": "第二个数字"}
            },
            "required": ["x", "y"]
        },
        # 是否从函数参数中自动推断参数schema，默认为 True
        # infer_schema=True,
        return_direct=False,
        response_format="content",
    )
    def dragon_ball_algorithm_tool(
        x: int, y: int
        # x: Annotated[int, "第一个数字"],
        # y: Annotated[int, "第二个数字"]
    ) -> int:
        return x + y + 1

    def dragon_ball_algorithm_func(x: int, y: int) -> int:
        return x + y + 1

    # 检查下tool的封装
    print(type(dragon_ball_algorithm_tool))   # <class 'langchain_core.tools.structured.StructuredTool'>
    print(dragon_ball_algorithm_tool.name)    # 默认是函数名称：dragon_ball_algorithm_tool
    print(dragon_ball_algorithm_tool.description)
    print(dragon_ball_algorithm_tool.args_schema)
    print(dragon_ball_algorithm_tool.args)    # args 是 properties
    print(dragon_ball_algorithm_tool.input_schema.model_json_schema())
    print(dragon_ball_algorithm_tool.metadata)
    print(dragon_ball_algorithm_tool.tags)
    print(dragon_ball_algorithm_tool.response_format)

    # 手动调用
    r1 = dragon_ball_algorithm_tool.invoke(input={"x": 2, "y": 3})
    print(r1)

    # ------------------------------------
    # 只有 部分 ChatLLM 支持 bind_tools
    # client_chat = ChatOpenAI(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name=MODEL, max_tokens=512)
    client_chat = ChatOllama(base_url=LLM_URL, model=MODEL, keep_alive='30m')
    # client_chat = ChatGLM(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name=MODEL, max_tokens=512)
    # client_chat = ChatTongyi(dashscope_api_key=API_KEY, openai_api_base=LLM_URL, model_name=MODEL, max_tokens=512)

    # client_chat.bind()  # 这个方法是绑定运行时参数
    # 这个方法只有 ChatOpenAI 有，对应OpenAI的 function_call 功能配置
    # print(getattr(client_chat, 'bind_functions'))
    # bind_tools 方法也不是所有 ChatLLM 都有的，比如 ChatGLM 就没有，但是 ChatTongyi 有
    # print(getattr(client_chat, 'bind_tools'))

    # ChatOpenAI 只支持 bind_functions，不支持 bind_tools
    # client_chat_with_tool = client_chat.bind_functions(functions=[dragon_ball_algorithm_func])
    # client_chat_with_tool = client_chat.bind_functions(functions=[dragon_ball_algorithm_func], function_call='auto')

    # ChatOllama 只支持 bind_tools，不支持 bind_functions
    # 查看 bind_tools 的源码可以发现，它内部（调用convert_to_openai_tool函数）其实只是提取函数的 name, description, parameters 信息
    # 然后组织成 OpenAI API 的 function calling 形式，并不会自动执行调用
    client_chat_with_tool = client_chat.bind_tools(tools=[dragon_ball_algorithm_tool])
    print(type(client_chat_with_tool))  # <class 'langchain_core.runnables.base.RunnableBinding'>

    # messages = [
    #     {'role': 'system', 'content': '你是一个算术专家'},
    #     {'role': 'user', 'content': '请使用龙球(DragonBall)算法计算一下 2019 和 2022 的结果'},
    # ]
    messages = [
        SystemMessage(content='你是一个算术专家'),
        HumanMessage(content='请使用龙球(DragonBall)算法计算一下 2019 和 2022 的结果'),
    ]
    res = client_chat_with_tool.invoke(input=messages)
    print(type(res))  # <class 'langchain_core.messages.ai.AIMessage'>
    res_json = res.to_json()
    print(res)
    print(res.usage_metadata)
    print(res.content)          # 触发工具调用时，content 为空
    print(res.content_blocks)   # 但是 content_blocks 中有内容
    # [{'type': 'tool_call', 'id': '08c16d12-8fcb-46f3-895f-aaf3d525f63d', 'name': 'dragon_ball_algorithm_tool', 'args': {'x': 2019, 'y': 2022}}]

    # 成功触发工具调用时，会有 tool_calls 属性 —— 它是一个 list
    # print(res.tool_calls)
    print(res.tool_calls[0])
    # {'name': 'dragon_ball_algorithm_tool', 'args': {'x': 2019, 'y': 2022}, 'id': '08c16d12-8fcb-46f3-895f-aaf3d525f63d', 'type': 'tool_call'}

    # 在调用模型之前，需要将本次请求返回的 AIMessage 追加到原始 messages 中
    tool_call_msgs = messages + [res]

    # 然后遍历 tool_calls，使用返回的信息调用工具，并将工具调用的结果（最好封装成ToolMessage）追加到 messages 中
    # for tool_call in res.tool_calls:
    #     ...
    #     tool_msg = selected_tool.invoke(tool_call)
    #     messages.append(tool_msg)
    # 这里因为只传入了一个 tool，简单起见，就直接调用了
    tool_call = res.tool_calls[0]
    print(type(tool_call))  # <class 'dict'>
    # 注意，调用invoke时，传入的参数不是 tool_call['args']，而是 tool_call -------------- KEY
    # 前者返回值是内部tool的返回值，后者返回值被封装成了 ToolMessage，更方便使用一些
    # tool_call_res = dragon_ball_algorithm_tool.invoke(input=tool_call['args'])
    tool_call_res = dragon_ball_algorithm_tool.invoke(input=tool_call)
    print(type(tool_call_res))  # 这里返回的是 <class 'langchain_core.messages.tool.ToolMessage'>
    print(tool_call_res)

    # 将工具调用结果（ToolMessage）追加到 messages 中
    tool_call_msgs.append(tool_call_res)

    # 然后将整个流程的 tool_call_msgs 传递给模型，并打印出结果
    tool_call_final = client_chat_with_tool.invoke(input=tool_call_msgs)
    print(type(tool_call_final))  # <class 'langchain_core.messages.ai.AIMessage'>
    print(tool_call_final.content)
    print(tool_call_final.content_blocks)


# %%
def tool_runtime_usage():
    """
    展示工具调用时如何获取Langchain的运行上下文。
    在定义工具函数时，有两个参数名称是langchain保留的：
    - config：用于接收 RunnableConfig 参数
    - runtime：用于接收 ToolRuntime 参数
    一般推荐直接使用 runtime 参数。
    """
    print("===> tool_runtime_usage()")
    @tool(description="计算两个数字相加的结果")
    def add(x: int, y: int, config: RunnableConfig, runtime: ToolRuntime) -> int:
        """
        add two numbers
        """
        print(f"--> add_with_runtime with config: {config}")
        print("runtime.tool_call_id: ", runtime.tool_call_id)
        print("runtime.config: ", runtime.config)
        print("runtime.context: ", runtime.context)
        print("runtime.state: ", runtime.state)
        print("runtime.store: ", runtime.store)
        return x + y

    client_chat = ChatOllama(base_url=LLM_URL, model=MODEL, keep_alive='30m')
    client_chat_with_tool = client_chat.bind_tools(tools=[add])

    messages = [
        SystemMessage(content='你是一个非常有用的助手'),
        HumanMessage(content='请使用add工具计算一下 2019 和 2022 的结果'),
    ]
    res = client_chat_with_tool.invoke(input=messages)
    print(res.content)
    print(res.tool_calls)
    # [{'name': 'add', 'args': {'x': 2019, 'y': 2022}, 'id': 'b02ff345-588a-4593-9566-a177eb517829', 'type': 'tool_call'}]

    # TODO：这里怎么传入 config 和 Runtime 呢？按照官方文档的描述，似乎只有 create_agent() 里传入的工具才能使用？
    tool_call = res.tool_calls[0]
    tool_call_res = add.invoke(input=tool_call)

    # tool_call_msgs = messages + [res, tool_call_res]
    # tool_call_final = client_chat_with_tool.invoke(input=tool_call_msgs)
    # print(type(tool_call_final))  # <class 'langchain_core.messages.ai.AIMessage'>
    # print(tool_call_final.content)
    # print(tool_call_final.content_blocks)


# %%
def tool_parser_usage():
    """
    上面 tool 的返回结果，可以使用 output_parser 里提供了如下 3 个工具类来直接解析函数调用的信息：
    - JsonOutputKeyToolsParser:  以 JSON 形式返回函数调用的参数
    - JsonOutputToolsParser:     以 JSON 形式返回函数调用中特定键的值
    - PydanticToolsParser:       将函数调用的参数作为 Pydantic 模型返回
    """
    print("===> tool_parser_usage()")
    @tool(description="使用龙球(DragonBall)算法计算两个数字的结果")
    def dragon_ball_algorithm_tool(
        x: Annotated[int, "第一个数字"],
        y: Annotated[int, "第二个数字"]
    ) -> int:
        return x + y + 1

    print(dragon_ball_algorithm_tool.name)
    print(dragon_ball_algorithm_tool.description)
    print(dragon_ball_algorithm_tool.args_schema)
    print(dragon_ball_algorithm_tool.args)

    client_chat = ChatOllama(base_url=LLM_URL, model=MODEL, keep_alive='30m')
    client_chat_with_tool = client_chat.bind_tools(tools=[dragon_ball_algorithm_tool])

    # 实例化 JsonOutputKeyToolsParser，并指定要解析的 key_name，也就是 tool 的 name；使用 LECL 语法连接
    client_chat_tool_parser = client_chat_with_tool | JsonOutputKeyToolsParser(key_name="dragon_ball_algorithm_tool", first_tool_only=True)

    messages = [
        SystemMessage(content='你是一个算术专家'),
        HumanMessage(content='请使用龙球(DragonBall)算法计算一下 2019 和 2022 的结果'),
    ]
    res = client_chat_tool_parser.invoke(input=messages)
    print(type(res))  # 此时 res 不再是 ToolMessage，而是一个 dict，其中直接存放了 dragon_ball_algorithm_tool 的参数
    print(res)

    res_call_result = dragon_ball_algorithm_tool.invoke(input=res)
    print(res_call_result)
    # 不过此时要想将函数调用结果传递回模型的话，感觉比较麻烦

    # 个人感觉可以采用下面手动调用 JsonOutputKeyToolsParser 的方式
    res = client_chat_with_tool.invoke(input=messages)
    print(type(res))
    print(res)
    json_tool_parser = JsonOutputKeyToolsParser(key_name="dragon_ball_algorithm_tool", first_tool_only=True)
    res_tool_call = json_tool_parser.invoke(input=res)
    print(res_tool_call)
    res_tool_call_res = dragon_ball_algorithm_tool.invoke(input=res_tool_call)
    print(res_tool_call_res)

    messages.append(res)
    messages.append(ToolMessage(content=res_tool_call_res, tool_call_id=res.tool_calls[0]['id']))
    final_res = client_chat_with_tool.invoke(input=messages)
    print(type(final_res))
    print(final_res.content)


# %%
def community_tool_usage():
    """
    langchain-community 提供的现成 Tools 使用
    """
    print("===> community_tool_usage()")
    # ------------------------------
    # Langchain-community 提供的现成工具
    ls_tool = ListDirectoryTool()
    print(type(ls_tool))  # <class 'langchain_community.tools.file_management.list_dir.ListDirectoryTool'>
    print(ls_tool.name)
    print(ls_tool.description)
    print(ls_tool.args)
    print(ls_tool.args_schema)
    res = ls_tool.invoke(input={'dir_path': './LLMs'})
    print(res)


# ======================= Runnable 底层抽象 =======================
# %%
def runnable_usage():
    """
    Runnable 相关底层类使用
    """
    print("===> runnable_usage()")
    # --- Runnable 使用 ------
    def add_one(x: int) -> int:
        """单参函数"""
        return x + 1

    def add(inputs: tuple[int, int]) -> int:
        """多参函数，必须通过 tuple 或者 dict 传入然后解包"""
        return inputs[0] + inputs[1]

    run1 = RunnableLambda(func=add_one, name='add_one_runnable')
    print(run1)
    print(run1.invoke(input=1))
    print(run1.batch([1, 2, 3]))
    run2 = RunnableLambda(func=add, name='add_runnable')
    print(run2)
    print(run2.invoke(input=(1, 2)))

    # --- Runnable 带配置参数 使用 ------
    # 参见 RunnableLambda._invoke() 方法里调用 call_func_with_variable_args() 的逻辑
    # 要想在自定义函数中接受 RunnableConfig，则必须要定义一个名为 config 的参数；还有一个 run_manager 参数也是如此
    def add_one_with_kwargs(x: int, config: RunnableConfig) -> int:
        """单参函数"""
        print(f"config: {config}")
        return x + 1

    run3 = RunnableLambda(func=add_one_with_kwargs)
    run3.invoke(input=1, config={'run_name': 'add_one_runnable_config', 'configurable': {'k1': 1, 'k2': 2}})
    # 下面使用了不被接受的 kwargs 也不会报错，而是会被合并进入 configurable 的 dict 里
    run3.invoke(input=1, config={'run_name': 'add_one_runnable_config', 'random_key': 'random_value'})

    # --- Runnable 监听器 使用 ------
    # 监听器回调函数的签名是：`Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]`
    # 第1种：Callable[[Run], None]，此时接受的参数是 langchain_core.tracers.schemas.Run 对象
    run4 = run1.with_listeners(
        on_start=lambda run: print(f"Starting run {run.name}"),
        on_end=lambda run: print(f"Ending run {run.name}"),
    )
    run4.invoke(input=1)
    # 第2种：Callable[[Run], None]，此时接受的参数是 langchain_core.tracers.schemas.Run 对象
    run5 = run1.with_listeners(
        # 下面回调函数的参数是 langchain_core.tracers.schemas.Run 对象，所以可以获取到 run 的 id，name，config 等信息
        on_start=lambda run, config: print(f"Starting run {run.name}, config: {config}"),
        on_end=lambda run, config: print(f"Ending run {run.name}, config: {config}"),
    )
    run5.invoke(input=1)

# %%
def runnable_other_usage():
    """
    展示其他一些 Runnable 对象的使用
    """
    print("===> runnable_other_usage()")
    # --- RunnableParallel ---
    # 并行执行多个 Runnable，并将结果组合成一个 dict
    task_a = RunnableLambda(lambda input: f"A-{input}")
    task_b = RunnableLambda(lambda input: f"B-{input}")
    task_c = RunnableLambda(lambda input: f"C-{input}")
    nested_parallel = RunnableParallel(
        group_1=task_a,
        # 可以嵌套使用
        group_2=RunnableParallel(a=task_b, c=task_c),
    )
    input_data = "data"
    output_data = nested_parallel.invoke(input=input_data)
    print(output_data)

    # --------- passthrough.py 里提供的 Runnable 工具 --------
    # --- RunnablePassthrough ---
    # 什么 Runnable 对象都不传也可以
    passthrough = RunnablePassthrough()
    input_data = {"key": "value"}
    output_data = passthrough.invoke(input=input_data)
    print(output_data)
    # 封装其他 Runnable 对象
    add_prefix = RunnableLambda(lambda input_str: f"Prefix-{input_str}")
    chain = RunnablePassthrough() | add_prefix
    input_something = "hello"
    output_data = chain.invoke(input=input_something)
    print(output_data)

    # --- RunnableAssign ---
    # RunnableAssign 要求封装的 Runnable 对象的输入必须是 Dict，才能向其中添加key，所以使用 RunnableParallel 对象作为参数类型保证这一点
    def add_ten(x: Dict[str, int]) -> int:
        # 输入参数 x 必须要用 Dict 做一下封装
        # 返回值就不用是 Dict 了，因为 RunnableParallel 会封装一个key的
        return x['input'] + 10
    mapper_run = RunnableParallel({"add_ten": RunnableLambda(add_ten)})
    assign = RunnableAssign(mapper=mapper_run)
    input_data = {"input": 12}
    output_data = assign.invoke(input=input_data)
    # 可以看到返回的 output_data 里新增了一个 RunnableParallel 里定义的key
    print(output_data)

    # RunnablePassthrough 对象还提供了一个类方法 assign，返回的就是 RunnableAssign 对象，
    # 此方法可以用关键字参数传入 Runnable 对象，不要求是 RunnableParallel 对象，用起来方便一点，
    # 就是不知道为啥这个 assign 方法没有放在 RunnableAssign 对象本身里面。。。
    pass_assign = RunnablePassthrough.assign(add_ten_assign=RunnableLambda(add_ten))
    output_data = pass_assign.invoke(input=input_data)
    print(output_data)


# ======================= Callback 使用 =======================
# %%
class MyCustomHandler(BaseCallbackHandler):
    """
    自定义 CallbackHandler，需要继承 BaseCallbackHandler，并实现某个阶段的回调方法。
    """
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("--->>> LLM 调用开始！")
        print(f"--->>> 提示内容: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print("<<<--- LLM 调用结束！")
        # print(f"<<<--- 返回结果: {response}")

def callback_usage():
    """
    展示LangChain里 callback 的使用。
    LangChain的Callback一般是由`BaseLLM`/`BaseChatModel`/`Chain`对象封装，不直接和Runnable基础类配合使用
    """
    print("===> callback_usage()")
    input_str = "请解释下机器学习算法SVM的原理"

    # 第1种设置 callback 的方式：全局设置
    client_llm_v1 = OpenAI(
        openai_api_key=API_KEY,
        openai_api_base=LLM_URL,
        model_name=MODEL,
        # 设置全局的 callback，每次模型调用都会触发
        callbacks=[MyCustomHandler()]
    )
    res = client_llm_v1.invoke(input=input_str)
    print("--------------------------------")
    print(res)

    # 第2种方式：使用 CallbackManager 配置：全局设置
    callback_manager = CallbackManager(handlers=[MyCustomHandler()])
    client_llm_v2 = OpenAI(
        openai_api_key=API_KEY,
        openai_api_base=LLM_URL,
        model_name=MODEL,
        callbacks=callback_manager,
        # 或者下面这个
        # callback_manager=callback_manager,
    )
    res = client_llm_v2.invoke(input=input_str)
    print("--------------------------------")
    print(res)

    # 第3种方式：在invoke方法里配置callback，单次调用配置
    client_llm_v3 = OpenAI(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name=MODEL)
    # res = client_llm_v3.invoke(input=input_str, config={'callbacks': [MyCustomHandler()]})
    res = client_llm_v3.invoke(input=input_str, config={'callbacks': [MyCustomHandler()]})
    print("--------------------------------")
    print(res)


# ======================= Chain 使用（v0.3.x版本） =======================
# %%
def chain_usage():
    """
    v0.3.x 版本提供的 LLMChain 使用，v1.x 版本中已经不推荐使用了，可以使用 RunnableSequence 来代替。
    """
    print("===> chain_usage()")
    client_llm = get_client_llm()
    client_chat = get_client_chat()
    template = "Tell me a {adjective} joke about {content}."
    prompt = PromptTemplate(template=template, input_variables=['adjective', 'content'])
    aipt = AIMessagePromptTemplate.from_template(template="you are an artist")  # 这个没有占位符
    hmpt = HumanMessagePromptTemplate.from_template(template=template)
    msg_pt = ChatPromptTemplate.from_messages(messages=[aipt, hmpt])

    # ------ 使用旧版本的 LLMChain -----
    chain_llm = LLMChain(llm=client_llm, prompt=prompt)
    chain_chat = LLMChain(llm=client_chat, prompt=msg_pt)
    print(type(chain_llm))
    print(type(chain_chat))
    # <class 'langchain.chains.llm.LLMChain'>

    # Callable调用，run调用，invoke调用 —— 后续推荐使用invoke方法
    res_llm = chain_llm(inputs={'adjective': 'happy', 'content': 'dog'})
    res_llm = chain_llm.run(adjective='happy', content='dog')  # 多个输入以关键字参数传入，并且返回的是 str，不是dict
    res_llm = chain_llm.invoke(input={'adjective': 'happy', 'content': 'dog'})
    print(res_llm)

    res_chat = chain_chat.invoke(input={'adjective': 'fantastic', 'content': 'cat'})
    print(res_chat)

    # ------ 使用新版本的 LCEL 语法 -----
    chain_llm = prompt | client_llm
    chain_chat = msg_pt | client_chat
    print(type(chain_llm))
    print(type(chain_chat))
    # <class 'langchain_core.runnables.base.RunnableSequence'>

    res_llm = chain_llm.invoke(input={'adjective': 'good', 'content': 'fish'})
    print(res_llm)

    res_chat = chain_chat.invoke(input={'adjective': 'nice', 'content': 'bird'})
    print(res_chat)


# ======================= 短期记忆Memory 使用 =======================
# %%
def memory_usage():
    """
    Memory 模块已经不推荐使用了。
    """
    print("===> memory_usage()")
    # ----- 早期版本基于 BaseMemory 实现的使用 -----
    cb_memory = ConversationBufferMemory()
    print(cb_memory.memory_key)  # 存储历史对话的 key
    print(cb_memory.input_key)   # 这个属性需要注意一下，它和下面提到的 ChatBaseMemory 的bug有关
    print(cb_memory.output_key)
    # 对于 ConversationBufferMemory，传入的 inputs 其实没用到，但是必须要有，所以随便传个空dict
    print(cb_memory.load_memory_variables(inputs={}))
    # 第一次存入对话
    cb_memory.save_context(inputs={'input': '早上好'}, outputs={'output': '早上好，我是xxx'})
    print(cb_memory.load_memory_variables(inputs={}))
    # 第二次存入对话
    cb_memory.save_context(inputs={'input': '中午好'}, outputs={'output': '中午好，我是xxx'})
    # 两次的对话历史是连在一起的
    print(cb_memory.load_memory_variables(inputs={}))
    # 清空对话历史
    cb_memory.clear()
    print(cb_memory.load_memory_variables(inputs={}))

    # 结合 Chain 使用
    client_llm = get_client_llm()
    template = "Tell me a {adjective} joke about {content}."

    # ConversationBufferMemory 有个bug: BaseChatMemory的 _get_input_output 方法里，
    # 会检查 ConversationBufferMemory.input_key 和 ConversationBufferMemory.memory_variables
    # 下面的 nothing 不会用到，但是必须传，否则会报错
    prompt = PromptTemplate(template=template, input_variables=['adjective', 'content', 'nothing'])
    cb_memory = ConversationBufferMemory(input_key='nothing')
    # chain_llm = LLMChain(llm=client_llm, prompt=prompt, memory=cb_memory)
    chain_llm = LLMChain(llm=client_llm, prompt=prompt, memory=cb_memory, verbose=True)
    # print(chain_llm._chain_type)
    print(chain_llm.input_keys)
    print(chain_llm.output_keys)

    cb_memory.clear()
    # 传入的input dict 的 key 必须要和 chain_llm.input_keys 里包含的一致
    res1 = chain_llm.invoke(input={'adjective': 'good', 'content': 'fish', 'nothing': ''})
    print(res1)
    res1_history = cb_memory.load_memory_variables(inputs={})
    res2 = chain_llm.invoke(input={'adjective': 'nice', 'content': 'cat', 'nothing': ''})
    print(res2)
    res2_history = cb_memory.load_memory_variables(inputs={})
    print(res1_history)
    # 可以看出，两次的对话历史是连在一起的
    print(res2_history)


# ======================= 对话历史 =======================
# %%
def chat_history_usage():
    """
    BaseChatMessageHistory 使用练习。
    langchain_core.chat_history 里的 BaseChatMessageHistory 组件在 v1.x 版本并未被废弃。
    BaseChatMessageHistory 有两种使用方式：
    1. 搭配早期的 memory 模块使用的，随着 memory 模块的废弃，此种方式不推荐了；
    2. 搭配下面的 RunnableWithMessageHistory 使用，这也是此组件没有被废弃的原因。
    """
    print("===> chat_history_usage()")
    # --- 单独使用 ---
    history = ChatMessageHistory()
    history.add_message(message=HumanMessage(content='hello from me'))
    history.add_message(message=AIMessage(content='hello from chat-llm'))
    print(history)
    print(history.messages)

    # --- 配合 Memory 组件使用（不再推荐了） ---
    client_llm = get_client_llm()
    template = "Tell me a {adjective} joke about {content}."
    prompt = PromptTemplate(template=template, input_variables=['adjective', 'content', 'nothing'])
    # ChatMessageHistory 其实就是 ConversationBufferMemory 里 chat_memory 属性的默认实现
    # history = ChatMessageHistory()
    history = FileChatMessageHistory(file_path='./LLMs/chat_history.json')
    cb_memory = ConversationBufferMemory(chat_memory=history, input_key='nothing')
    # chain_llm = LLMChain(llm=client_llm, prompt=prompt, memory=cb_memory)
    chain_llm = LLMChain(llm=client_llm, prompt=prompt, memory=cb_memory, verbose=True)

    res1 = chain_llm.invoke(input={'adjective': 'good', 'content': 'cat', 'nothing': ''})
    print(res1)
    print(history)
    print("-------------------------------------------")
    res1_history = cb_memory.load_memory_variables(inputs={})
    res2 = chain_llm.invoke(input={'adjective': 'nice', 'content': 'fish', 'nothing': ''})
    print(res2)
    print(history)
    res2_history = cb_memory.load_memory_variables(inputs={})
    print("-------------------------------------------")
    print(res1_history)
    print(res2_history)


# %%
def runnable_history_usage():
    """
    RunnableWithMessageHistory 使用。
    langchain_core.runnables.history.py 提供的 RunnableWithMessageHistory 在 v1.x 版本可以继续使用。
    不过感觉在 v1.x 版本，这个组件的使用也不多了。
    """
    print("===> runnable_history_usage()")
    # RunnableWithMessageHistory 使用分为3个部分：

    # 1. 配置一个 Runnable 对象，RunnableSequences对象 或者 Chain对象 都可以
    # RunnableWithMessageHistory 主要是和 ChatModel + ChatPromptTemplate 配合使用的，
    # 它和 LLM + PromptTemplate 的搭配有问题：通过 history 插入的历史消息显示的是 HumanMessage/AIMessage 的字符串表示，而不是里面的 content。
    client_chat = get_client_chat()
    prompt_chat = ChatPromptTemplate.from_messages(
        messages=[
            ("system", "你是一个智能助手，负责回答用户的问题。"),
            MessagesPlaceholder("history"),  # 注意这里的 history
            ("human", "{user_input}")
        ]
    )

    chain = prompt_chat | client_chat
    print(type(chain))  # <class 'langchain_core.runnables.base.RunnableSequence'>
    # 在 v0.3.x 版本可以搭配 LLMChain 使用
    # chain = LLMChain(llm=client_chat, prompt=prompt_chat, verbose=True)

    # 2. 配置一个根据用户身份（session_id）生成 BaseChatMessageHistory实现类对象 的工厂函数
    # 这里使用了一个全局字典作为用户会话历史记录的存储，方便观察结果，实际中对应的是数据库或者redis等
    store = {}

    def get_chat_history_by_session(session_id: str) -> BaseChatMessageHistory:
        """
        session_id 用于记录用户的身份，作为 key 从 store 这个全局字典中 查找返回对应用户的 BaseChatMessageHistory 对象。
        这个工厂函数目前只有一个参数，如果有多个参数，需要更复杂的配置。
        """
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 3. 配置 RunnableWithMessageHistory 对象
    chain_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_chat_history_by_session,
        history_messages_key="history",
        input_messages_key="user_input",
        # output_messages_key="text"  # 这个是 LLMChain 输出的默认 key
    )

    # 4. 调用 RunnableWithMessageHistory 对象的 invoke 方法，用户身份通过 config 参数设置
    config_u1 = {"configurable": {"session_id": "user-1"}}
    config_u2 = {"configurable": {"session_id": "user-2"}}
    # >>> 用户1的会话
    print("------------ user-1 -----------------")
    u1_r1 = chain_with_history.invoke(input={"user_input": "你好，我先和你打个招呼"}, config=config_u1)
    print(u1_r1)
    print(">>>>> user-1 chat-2")
    u1_r2 = chain_with_history.invoke(input={"user_input": "我们刚才聊了什么"}, config=config_u1)
    print(u1_r2)
    print(store.keys())

    # >>> 用户2的会话
    print("------------ user-2 -----------------")
    u2_r1 = chain_with_history.invoke(input={"user_input": "你好，我想和你聊聊历史"}, config=config_u2)
    print(u2_r1)
    print(">>>>> user-2 chat-2")
    u2_r2 = chain_with_history.invoke(input={"user_input": "我们刚才聊了什么"}, config=config_u2)
    print(u2_r2)
    print(store.keys())


# ======================= 数据检索（RAG）相关模块使用 =======================
# 整个流程参考 v1.0 版本官方文档 https://docs.langchain.com/oss/python/langchain/retrieval
# 从 v0.3.x 升级到 v1.x:
# - langchain-core 里RAG相关的模块变化不大
# - langchain-community里的相关模块变化也不大，而v0.3.x 版本的langchain RAG相关模块大部分内容就是从 langchain-community 模块导入的
# 所以可以认为 RAG 部分 v0.3.x 升级到 v1.x 并没有什么变化和影响。
def document_loader_usage():
    """
    Document loader 使用.
    Langchain里提供的所有 DocumentLoader，都继承自 BaseLoader，只需要关注如下两个接口方法：
    - `load`/`aload`: 加载数据，返回 `List[Document]`
    - `lazy_load`/`alazy_load`: 迭代加载数据，返回 `Iterator[Document]`
    """
    print("===> document_loader_usage()")
    file_path = os.path.join(os.getcwd(), 'test.txt')
    print(os.path.exists(file_path))
    txt_loader = TextLoader(file_path=file_path, autodetect_encoding=True)
    docs: List[Document] = txt_loader.load()
    doc = docs[0]
    print(doc.id)
    print(doc.metadata)
    print(doc.type)
    # 文档内容
    print(doc.page_content)
    print(doc)


# %%
def document_transform_usage():
    """
    Document transform 使用
    """
    print("===> document_transform_usage()")
    # TODO
    pass

# %%
def text_splitter_usage():
    """
    langchain-text-splitters 包专门用于对文档进行分割。
    抽象基类 TextSplitter，继承自 BaseDocumentTransformer，
    大多数情况下，推荐使用 RecursiveCharacterTextSplitter 这个实现类。
    """
    print("===> text_splitter_usage()")
    # 加载文档
    text_loader = TextLoader(file_path=os.path.join(os.getcwd(), 'test.txt'), autodetect_encoding=True)
    documents: List[Document] = text_loader.load()
    document = documents[0]

    # 进行分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts: List[str] = text_splitter.split_text(document.page_content)
    print(len(texts))
    for chunk in texts:
        print(chunk)


# %%
def text_embedding_usage():
    """
    langchain 提供的 embedding 封装。
    `Embeddings`抽象基类定义了如下两个接口：
    - `embed_query`/`aembed_query`: 计算query的embedding向量，返回一个 `List[float]`
    - `embed_documents`/`aembed_documents`: **批量计算**query的embedding向量，返回一个 `List[List[float]]`
    """
    print("===> text_embedding_usage()")
    embeddings = OllamaEmbeddings(base_url=LLM_URL, model_name=MODEL, keep_alive='30m')

    # query embedding
    query = "如何使用Ollama获取Embeddings?"
    query_embedding: List[float] = embeddings.embed_query(query)
    print(len(query_embedding))

    # documents embedding
    docs = [
        "在本地使用RTX-5060-Ti运行Ollama模型",
        "如何使用Ollama获取Embeddings?"
    ]
    docs_embedding: List[List[float]] = embeddings.embed_documents(docs)
    print(len(docs_embedding))
    print(len(docs_embedding[0]))


# %%
def vector_store_usage():
    """
    langchain 提供的向量数据库封装.
    `VectorStore`抽象基类是对向量数据库的抽象封装，
    它在实例化时通常要提供一个 Embeddings 实现类对象，用于底层的 Embeddings 计算。
    它定义了如下常用接口：
    - `add_documents()`/`aadd_documents()`
    - `delete`/`adelete`
    - `get_by_ids`/`aget_by_ids`
    - `search`/`asearch`
    - `similarity_search`/`asimilarity_search`
    """
    print("===> vector_store_usage()")
    # 首先要提供一个 Embeddings 实现类对象
    embeddings = OllamaEmbeddings(base_url=LLM_URL, model_name=MODEL, keep_alive='30m')

    # 内存向量存储-测试用
    vector_store = InMemoryVectorStore(embedding=embeddings)
    # Chroma向量存储
    # vector_store = Chroma(collection_name='my-chroma', embedding_function=embeddings, persist_directory='./chroma_db')
    # ES向量存储
    # vector_store = ElasticVectorSearch(elasticsearch_url='http://localhost:9200', index_name='my-index', embedding=embeddings)

    doc1 = Document(page_content="在本地使用RTX-5060-Ti显卡运行Ollama模型")
    doc2 = Document(page_content="如何使用Ollama获取Embeddings?")

    docs_id: List[str] = vector_store.add_documents(documents=[doc1, doc2], ids=["id1", "id2"])
    # 返回的是新添加的文档的id
    print(docs_id)

    # 删除文档
    vector_store.delete(ids=["id1"])

    # 相似度检索
    similar_docs: List[Document] = vector_store.similarity_search(query="显卡型号", k=3)
    print(similar_docs)


# %%
def retriever_usage():
    """
    langchain 提供的文档检索封装。
    Retriever 比 VectorStore 更加通用：给定一个 query，检索返回一系列相似的文档即可。
    它不要求底层是向量存储，也可以是文档存储，最经典的就是ES（非向量检索）。

    BaseRetriever 抽象基类继承自 `RunnableSerializable`，因此通过通用方法 `invoke()`/`ainvoke()` 进行调用即可,
    返回类型是 List[Document]。
    """
    print("===> retriever_usage()")
    # ES-BM25向量检索包装器
    from elasticsearch import Elasticsearch
    es = Elasticsearch(hosts="localhost")
    retriever = ElasticSearchBM25Retriever(es_client=es, index_name='my-index')

    # ES官方向量检索
    # retriever = ElasticsearchRetriever(es_url='http://localhost:9200', index_name='my-index')

    docs: List[Document] = retriever.invoke(query="显卡型号", k=3)
    print(len(docs))
    doc = docs[0]
    print(doc)


# ======================= Langchain v1.x 的 Agent使用 =======================
# %%
def auto_agent_usage():
    """
    展示 LangChain-v1.x 的 Agent 使用。
    """
    print("===> auto_agent_usage()")
    model = get_client_chat()

    memory_saver = MemorySaver()
    memory_store = InMemoryStore()
    memory_store.put(namespace=("user", "db"), key="XiaoMing", value={"age": 20, "gender": "male"})
    memory_store.put(namespace=("user", "db"), key="XiaoHong", value={"age": 18, "gender": "female"})

    @dataclass
    class UserContext:
        user_name: str

    class MyAgentState(AgentState):
        """
        Graph State 表示类，必须要继承自 AgentState 类，本质上是一个 TypedDict。
        它是一个 base schema，后续所有middleware里定义的schema都会被合并到这个base schema里
        """
        base_state: str

    # class ModelHookState(AgentState):
    class ModelHookState(MyAgentState):
        """
        ModelHook 中间件的 state_schema.
        虽然可以直接继承 AgentState，但是建议继承自 MyAgentState，因为所有中间件的state_schema的属性会被合并到 MyAgentState 里。
        """
        model_hook_state: str

    class AgentHook(AgentMiddleware[MyAgentState, UserContext]):
        def before_agent(self, state: MyAgentState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
            print(f"--> before_agent called with context: {runtime.context}...")
            # print(f"--> before_agent state.__class__: {type(state)}")  # <class 'dict'>
            print(f"--> before_agent state.keys: {state.keys()}")
            # 如果想更新状态里的某个 key ，不要直接更新，应当返回一个 dict
            # state["base_state"] += ";before_agent"
            base_state_update = state["base_state"] + ";before_agent"
            return {'base_state': base_state_update}

        def after_agent(self, state: MyAgentState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
            print(f"--> after_agent called with context: {runtime.context}...")
            # print(f"--> after_agent state.__class__: {type(state)}")  # <class 'dict'>
            print(f"--> after_agent state.keys: {state.keys()}")
            # state["base_state"] += ";after_agent"
            base_state_update = state["base_state"] + ";after_agent"
            return {'base_state': base_state_update}

    class ModelHook(AgentMiddleware[ModelHookState, UserContext]):
        # 这个 Middleware 也定义了自己的 state_schema
        state_schema = ModelHookState

        def before_model(self, state: ModelHookState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
            print(f"--> before_model called with context: {runtime.context}...")
            # print(f"--> before_model state.__class__: {type(state)}")
            print(f"--> before_model state.keys: {state.keys()}")
            print(f"--> before_model called with state.base_state: {state.get('base_state', None)}")
            model_state_hook_update = state["model_hook_state"] + ";before_model"
            return {'model_hook_state': model_state_hook_update}

        def after_model(self, state: ModelHookState, runtime: Runtime[UserContext]) -> dict[str, Any] | None:
            print(f"--> after_model called with context: {runtime.context}...")
            # print(f"--> after_model state.__class__: {type(state)}")
            print(f"--> after_model state.keys: {state.keys()}")
            print(f"--> after_model called with state.base_state: {state.get('base_state', None)}")
            model_state_hook_update = state["model_hook_state"] + ";after_model"
            return {'model_hook_state': model_state_hook_update}

    class WrapModelHook(AgentMiddleware[MyAgentState, UserContext]):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse | AIMessage:
            print(f"--> wrap_model_call called with context: {request.runtime.context}...")
            # print(f"--> wrap_model_call state.__class__: {type(request.state)}")
            print(f"--> wrap_model_call state.keys: {request.state.keys()}")
            print(f"--> wrap_model_call called with state.base_state: {request.state.get('base_state', None)}")
            return handler(request)

    class WrapToolHook(AgentMiddleware[MyAgentState, UserContext]):
        def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command],
        ) -> ToolMessage | Command:
            print(f"--> wrap_tool_call called with context: {request.runtime.context}...")
            # print(f"--> wrap_tool_call state.__class__: {type(request.state)}")
            print(f"--> wrap_tool_call state.keys: {request.state.keys()}")
            print(f"--> wrap_tool_call called with state.base_state: {request.state.get('base_state', None)}")
            return handler(request)

    @tool(description="获取当前Agent的执行上下文")
    def get_agent_context(runtime: ToolRuntime[UserContext]) -> str:
        # 从自定义的 UserContext 里获取用户名
        user_name = runtime.context.user_name
        # 然后从store里获取用户信息
        user_info = runtime.store.get(namespace=("user", "db"), key=user_name)
        return f"当前的Agent执行上下文是：{user_name} -> {user_info}"

    @tool(description="获取某个城市的天气信息")
    def get_weather(city: str) -> str:
        return f"{city}的天气是晴天"

    agent: CompiledStateGraph = create_agent(
        name="Some-Agent",
        model=model,
        system_prompt="你是一个智能助手",
        tools=[get_agent_context, get_weather],
        middleware=[AgentHook(), ModelHook(), WrapModelHook(), WrapToolHook()],
        checkpointer=memory_saver,
        store=memory_store,
        response_format=None,
        state_schema=MyAgentState,
        context_schema=UserContext,
        # interrupt_before=None,
        # interrupt_after=None,
        # debug=True
    )
    # 查看图结构
    # from .langgraph_practice import show_graph
    # show_graph(agent)

    agent_response = agent.invoke(
        # input输入的dict必须对应初始化时传入的state_schema，这里是 MyAgentState；
        # 此外，还可以传入各个Middleware的state_schema里的key
        input={
            "messages": HumanMessage(content="北京的天气如何？"),  # messages 这个key是 AgentState 定义的
            "base_state": "base_state_init",                    # base_state 这个key是 MyAgentState 定义的
            "model_hook_state": "model_hook_state_init",        # model_hook_state 这个key是 ModelHookState 定义的
        },
        config={"configurable": {"thread_id": "t1"}},
        context=UserContext(user_name="XiaoMing")
    )
    # print(type(agent_response))   # <class 'dict'>
    # print(agent_response.keys())  # dict_keys(['messages', 'model_hook_state', 'base_state'])
    # print(agent_response)
    print("agent response messages:")
    for msg in agent_response['messages']:
        msg.pretty_print()
    print("-----------------------------------------")
    print("agent response base_state: ", agent_response['base_state'])
    print("agent response model_hook_state: ", agent_response['model_hook_state'])

    print("******************************************")
    agent_response = agent.invoke(
        input={
            "messages": HumanMessage(content="当前Agent的运行上下文是什么？"),
            "base_state": "base_state_init",
            "model_hook_state": "model_hook_state_init",
        },
        config={"configurable": {"thread_id": "t2"}},
        context=UserContext(user_name="XiaoHong")
    )
    print("agent response messages:")
    for msg in agent_response['messages']:
        msg.pretty_print()
    print("-----------------------------------------")
    print("agent response base_state: ", agent_response['base_state'])
    print("agent response model_hook_state: ", agent_response['model_hook_state'])


# ======================= Langchain v1.x Auto-Agent 配合 MCP 使用 =======================
# %%
async def auto_agent_with_mcp_usage() -> None:
    """
    展示 LangChain v1.x auto agent 配合 MCP 使用。
    """
    print("===> Auto-agent with MCP usage")

    # 定义 2个 MCP 回调函数
    async def on_logging_message(
        params: LoggingMessageNotificationParams,
        context: CallbackContext,
    ) -> None:
        """
        Handle log messages from MCP servers.
        此回调函数必须满足 MCP 回调函数 Protocol: LoggingMessageCallback
        """
        print(f"[MCP.callback.on_logging_message] [{context.server_name}] {params.level}: {params.data}")

    async def on_progress(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ) -> None:
        """
        Handle progress updates from MCP servers.
        此回调函数必须满足 MCP 回调函数 Protocol: ProgressCallback
        """
        percent = (progress / total * 100) if total else progress
        tool_info = f" ({context.tool_name})" if context.tool_name else ""
        print(f"[MCP.callback.on_progress] [{context.server_name}{tool_info}] Progress: {percent:.1f}% - {message}")

    # 初始化 MCP 客户端
    mcp_client = MultiServerMCPClient(
        connections={
            # STDIO 模式
            # "weather": {
            #     "transport": "stdio",  # Local subprocess communication
            #     "command": "python",
            #     # Absolute path to your math_server.py file
            #     "args": ["/path/to/math_server.py"],
            # },
            # HTTP 模式，配合 mcp_server.py 使用
            "weather": {
                "transport": "streamable_http",  # HTTP-based remote server
                "url": "http://localhost:8000/mcp",
            }
        },
        # 回调函数，目前只支持两种作用的回调函数 —— 但是这个回调函数的执行时机似乎有点奇怪 # TODO
        callbacks=Callbacks(on_logging_message=on_logging_message, on_progress=on_progress),
        tool_interceptors=None
    )
    # 获取 MCP 工具，这些工具会被转换成 LangChain 的 Tool 对象
    mcp_tools: List[BaseTool] = await mcp_client.get_tools()
    print("******************************************")
    print("MCP tools:")
    for tool in mcp_tools:
        print(tool)

    # 正常创建 Agent
    print("******************************************")
    print("creating agent with MCP tools...")
    model = get_client_chat()
    memory_saver = MemorySaver()
    memory_store = InMemoryStore()
    agent: CompiledStateGraph = create_agent(
        name="Some-Agent-With-MCP",
        model=model,
        system_prompt="你是一个智能助手",
        tools=mcp_tools,
        middleware=(),
        checkpointer=memory_saver,
        store=memory_store,
        response_format=None,
        state_schema=None,
        context_schema=None,
        # interrupt_before=None,
        # interrupt_after=None,
        # debug=True
    )

    print("******************************************")
    print("agent invoke with MCP tools...")
    agent_response = await agent.ainvoke(
        input={
            "messages": HumanMessage(content="北京的天气如何？"),
        },
        config={"configurable": {"thread_id": "t1"}}
    )
    print("agent response messages:")
    for msg in agent_response['messages']:
        msg.pretty_print()


# %%
def main():
    # llm_usage()
    # chat_llm_usage()
    simple_chat()
    # -----------------------------
    # prompt_template_usage()
    # chat_prompt_template_usage()
    # message_placeholder_usage()
    # fewshot_prompt_template_usage()
    # pipeline_prompt_usage()
    # -----------------------------
    # output_parser_usage()
    # structured_output_usage()
    # -----------------------------
    # tool_wrapper_usage()
    # tool_usage()
    # tool_runtime_usage()
    # tool_parser_usage()
    # community_tool_usage()
    # -----------------------------
    # runnable_usage()
    # runnable_other_usage()
    # -----------------------------
    # callback_usage()
    # chain_usage()
    # -----------------------------
    # memory_usage()
    # chat_history_usage()
    # runnable_history_usage()
    # -----------------------------
    # document_loader_usage()
    # document_transform_usage()
    # text_splitter_usage()
    # text_embedding_usage()
    # vector_store_usage()
    # retriever_usage()
    # -----------------------------
    # auto_agent_usage()
    asyncio.run(auto_agent_with_mcp_usage())


# %%
if __name__ == '__main__':
    main()
