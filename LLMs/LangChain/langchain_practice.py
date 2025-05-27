import os
from typing import Optional, Dict, List, Union
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field
# --- 模型包装器的基类 ---
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.llms import BaseLLM, LLM
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
# --- LLM 模型包装器 ---
# from langchain.llms import OpenAI, ChatGLM, Tongyi, Ollama, VLLM  # 这个用法过时了，它只是从下面的 langchain_community.llms 中导入对应对象
# from langchain_community.llms import OpenAI, ChatGLM, Tongyi, Ollama, VLLM
# 上面的导入其实是从下面位置导入的包装器对象
# from langchain_community.llms.openai import OpenAI
# 不过对于 OpenAI 客户端，官方文档又建议后续直接从下面单独的 langchain-openai 包里导入
from langchain_openai.llms import OpenAI
# from langchain_community.llms.ollama import Ollama
from langchain_ollama.llms import OllamaLLM
from langchain_community.llms.vllm import VLLM
from langchain_community.llms.tongyi import Tongyi
from langchain_community.llms.chatglm import ChatGLM
# --- ChatLLM 模型包装器 ---
from langchain.chat_models import init_chat_model    # 模型初始化函数，根据名称自动选择模型
# from langchain_community.chat_models import ChatOpenAI, ChatOllama
# 对于 ChatOpenAI、ChatOllama，官方文档建议后续从 langchain_openai、langchain_ollama 包中导入
from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp, ChatTongyi, ChatHuggingFace
# ----------
from langchain_core.messages import ChatMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage, FunctionMessage
from langchain_core.prompts import StringPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder, ChatMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, FewShotChatMessagePromptTemplate, PipelinePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser, MarkdownListOutputParser
from langchain_core.output_parsers import JsonOutputKeyToolsParser, JsonOutputToolsParser, PydanticToolsParser
# ----------
# document_loaders, embeddings, vectorstores, retrievers 都是 langchain_community 包里的内容，官方建议直接从langchain_community包中导入
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, CSVLoader, JSONLoader, WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS, Cassandra, Clickhouse, Milvus, OpenSearchVectorSearch, \
#     SKLearnVectorStore, ElasticsearchStore, ElasticVectorSearch, ElasticKnnSearch
# from langchain_community.retrievers import BM25Retriever, ElasticSearchBM25Retriever
# ----------
from langchain_core.tracers.schemas import Run
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSequence, RunnableBinding, RunnableParallel
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager, StdOutCallbackHandler
from langchain_core.runnables.passthrough import RunnablePassthrough, RunnableAssign, RunnablePick
# ----------
from langchain.chains.llm import LLMChain
from langchain_core.memory import BaseMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory
# from langchain.memory import ChatMessageHistory, FileChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
# ----------
from langchain_core.tools import BaseTool, BaseToolkit, Tool, StructuredTool, tool
# from langchain.tools import ListDirectoryTool, ReadFileTool, WriteFileTool, HumanInputRun, ShellTool
from langchain_community.tools import ListDirectoryTool, ReadFileTool, WriteFileTool, HumanInputRun, ShellTool
# ----------
from langchain.globals import set_verbose
from langchain.callbacks.tracers import ConsoleCallbackHandler

# --- vLLM 部署 ---
# API_KEY = 'Empty'
# LLM_URL = 'http://172.16.0.32:10086/v1'
# MODEL = 'Qwen2.5-32B-Instruct'
# --- Ollama 本地部署 ---
API_KEY = 'Empty'
LLM_URL = 'http://localhost:11434'
# MODEL = 'qwen2.5:7b'
MODEL = 'qwen3:8b'

# ======================= LLM + ChatLLM 模型包装器 使用 =======================
def llm_usage():
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
    res = client_llm.invoke(input=input_str)
    print(res)

    for res in client_llm.stream(input=input_str):
        print(res, end='')

    inputs = ["请解释下机器学习算法SVM的原理", "请解释下机器学习算法GBDT的原理"]
    res = client_llm.batch(inputs=inputs)
    print(res[0])
    print(res[1])

    # 也可以直接使用 __call__ 方法，这是 BaseLLM/BastChatModel 提供的Callable调用，不过后续版本可能会移除此种调用方式
    res = client_llm(prompt=input_str)
    print(res)


def chat_llm_usage():
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
        {'role': 'system', 'content': '你是一个机器学习方面的专家'},
        {'role': 'user', 'content': '请问什么是SVM算法'},
    ]
    res = client_chat.invoke(input=messages)
    # print(res)
    print(res.content)

    for res in client_chat.stream(input=messages):
        print(res.content, end='')

    # Callable调用，不过只支持 List[BaseMessage] 参数，并且被标记为deprecated
    msg = [SystemMessage(content='你是一个机器学习方面的专家'), HumanMessage(content='请问什么是SVM算法')]
    res = client_chat(messages=msg)
    print(res.content)


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
        keep_alive='30m'
    )
    return client_llm

def get_client_chat() -> Union[BaseChatModel, SimpleChatModel]:
    # client_chat = ChatOpenAI(
    #     openai_api_key=API_KEY,
    #     openai_api_base=LLM_URL,
    #     model_name=MODEL,
    #     # temperature=0.7,
    #     # top_p=1,
    #     # streaming=False,
    # )
    client_chat = ChatOllama(
        base_url=LLM_URL,
        model=MODEL,
        # temperature=0.7,
        # top_p=1,
        keep_alive='30m'
    )
    return client_chat

# ======================= PromptTemplate + Message 使用 =======================
def prompt_template_usage():
    # StringPromptTemplate含有抽象方法，不能实例化
    # pt = StringPromptTemplate(input_variables=["p1", "p2"], template="content-1: {p1}, content-2: {p2}")

    # PromptTemplate 是 Completion 模型使用的基础模版
    template = "Tell me a {adjective} joke about {content}."

    # 第1种：直接实例化
    pt1 = PromptTemplate(input_variables=["adjective", "content"], template=template)
    pt1.format(adjective="funny", content="chickens")

    # 第2种：使用类方法 from_template —— 推荐这种方式
    pt2 = PromptTemplate.from_template(template=template)
    pt2.format(adjective="nice", content="dog")
    print(pt2.template)
    print(pt2.template_format)
    print(pt2.input_variables)

    # 下面返回的是 <class 'langchain_core.prompt_values.StringPromptValue'>
    pv2 = pt2.format_prompt(adjective="fantastic", content="fish")
    print(type(pv2))
    print(pv2)


def message_usage():
    """
    官方文档[Messages](https://python.langchain.com/docs/concepts/messages/)
    展示各类 Message 封装类的使用
    """
    # ----- ChatMessage 使用 -----
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

    # ----- SystemMessage/HumanMessage/AIMessage 等 使用 -----
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
    # ----- ChatMessagePromptTemplate 使用 -----
    template1 = "Tell me a {adjective} joke about {content}."
    # ChatMessagePromptTemplate 必须要指定 template 和 role
    cmpt = ChatMessagePromptTemplate.from_template(template=template1, role="user")
    # format 方法返回的是 ChatMessage 对象
    cmpt_msg = cmpt.format(adjective="nice", content="fish")
    print(type(cmpt_msg))  # <class 'langchain_core.messages.chat.ChatMessage'>
    print(cmpt_msg)
    print(cmpt_msg.content)
    print(cmpt_msg.type)  # chat
    print(cmpt_msg.role)  # user
    print(cmpt_msg)  # content='Tell me a nice joke about fish.' additional_kwargs={} response_metadata={} role='user'
    print(cmpt_msg.json())
    print(cmpt_msg.pretty_repr())
    cmpt_msg.pretty_print()
    # 还有一个 format_messages 方法，返回的是 List[ChatMessage]
    cmpt_msgs = cmpt.format_messages(adjective="nice", content="cat")
    print(type(cmpt_msgs))     # <class 'list'>
    print(type(cmpt_msgs[0]))  # <class 'langchain_core.messages.chat.ChatMessage'>

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
    # 使用 __init__ 方法或者 from_messages() 方法实例化对象，实际上 from_messages() 方法底层就是直接调用的 __init__() 方法，
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


def placeholder_usage():
    # ------ MessagesPlaceholder 使用 ------
    # 注意，MessagesPlaceholder 只能用于 ChatPromptTemplate 中，不能搭配 PromptTemplate 使用
    # 使用 optional=True，表示这个变量是可选的，如果不传，则不会报错，但会返回空列表
    prompt = MessagesPlaceholder(variable_name="history", optional=True)
    # 如果没有 optional=True，下面会抛异常
    print(prompt.format_messages())
    # 传入一系列消息
    history = [("system", "You are an AI assistant."), HumanMessage(content="Hello!")]
    res = prompt.format_messages(history=history)
    print(res)
    for msg in res:
        print(msg.content)

    # 组合 MessagesPlaceholder + ChatPromptTemplate 使用，构造对话历史模版
    chat_prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", "你是一个智能助手，负责回答用户的问题。"),
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


def fewshot_prompt_template_usage():
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

def pipeline_prompt_usage():
    """
    PipelinePromptTemplate 被标识为 Deprecated 了，所以不做研究
    """
    ...


def simple_chat():
    client_chat = get_client_chat()
    msg = [
        HumanMessage(content='RTX 5060 Ti 16GB跑本地大模型怎么样？'),
    ]
    res = client_chat.invoke(input=msg)
    print(res)
    # for chunk in client_chat.stream(input=msg):
    #     print(chunk.content, end='')


# ======================= Output Parser 使用 =======================
# LangChain的输出解析器是和提示词配合使用的，它会在提示词的末尾增加一段要求大模型输出指定格式的指令。
def output_parser_usage():
    """
    常用的有如下几种Parser:
    - StrOutputParser: 原样返回
    - JsonOutputParser: 以JSON格式返回
    - PydanticOutputParser: 以Pydantic对象返回，它继承自JsonOutputParser
    - MarkdownListOutputParser: 以MarkDown列表形式返回
    """
    class MyModel(BaseModel):
        name: str
        age: int
        position: str
        achievements: List[str]
    # 先实例化一个 parser 对象，可以通过 get_format_instructions 查看该Parser的格式化提示词指令
    # parser = StrOutputParser() # 注意，StrOutputParser没有提示词，因为它原样输出
    # parser = JsonOutputParser(pydantic_object=MyModel)
    parser = PydanticOutputParser(pydantic_object=MyModel)
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


def structured_output_usage():
    """
    展示如何输出结构化的内容，参考官方文档：
    [How-to-Guides -> How to return structured data from a model](https://python.langchain.com/docs/how_to/structured_output/)
    这里主要是使用 BaseLanguageModel 定义的抽象方法 with_structured_output，该方法接受一个 schema，用于描述模型输出的字段，返回一个 Runnable 对象。
    一般有 3 种指定 schema 的方式：
    1. TypedDict
    2. JSON Schema
    3. Pydantic Model
    使用 1和2 返回的是一个 dict，使用 3 返回的是对应 Pydantic Model 的对象
    with_structured_output 方法的实现交给了具体的模型类，但不是所有的模型类都实现了此方法。
    **一般只有 ChatModel 有此方法，因为 BaseChatModel 提供了一个默认实现**。
    具体有哪些模型实现了， 可以参考 https://python.langchain.com/docs/integrations/chat/#featured-providers 表格。
    """

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
    print(getattr(client_chat, 'with_structured_output'))
    print(getattr(client_chat, 'bind_tools'))
    # 如果使用 ChatOllama，不要使用从 langchain_community.chat_models 导入的 ChatOllama，
    # 而是使用 langchain_ollama.chat_models 里的 ChatOllama，因为前者没有实现自己的 bind_tools 方法，会报错

    structured_chat_dict = client_chat.with_structured_output(schema=JokeDict)
    # structured_chat_dict = client_chat.with_structured_output(schema=JokeDict, include_raw=True)
    res_dict = structured_chat_dict.invoke("Tell me a joke about cats")
    print(type(res_dict))
    print(res_dict)

    structured_chat_model = client_chat.with_structured_output(schema=JokeModel)
    # 如果使用了 include_raw=True，那么返回的 res_model 是一个 dict，而不是一个 Pydantic Model
    # structured_chat_model = client_chat.with_structured_output(schema=JokeModel, include_raw=True)
    res_model = structured_chat_model.invoke("Tell me a joke about cats")
    print(type(res_model))
    print(res_model)


# ======================= 数据检索相关模块使用 =======================
def document_loader_usage():
    file_path = os.path.join(os.getcwd(), 'test.txt')
    print(os.path.exists(file_path))
    txt_loader = TextLoader(file_path=file_path, autodetect_encoding=True)
    docs = txt_loader.load()
    doc = docs[0]
    print(doc.id)
    print(doc.metadata)
    print(doc.type)
    print(doc.page_content)
    print(doc)


def text_embedding_usage():
    # TODO
    pass


def vector_store_usage():
    # TODO
    pass


def retriever_usage():
    # TODO
    pass


# ======================= Chain 相关模块使用 =======================
def runnable_usage():
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


def runnable_other_usage():
    # 展示其他一些 Runnable 对象的使用
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


def chain_usage():
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

    # Callable调用，run调用，invoke调用——后续推荐使用invoke方法
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


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("--->>> LLM 调用开始！")
        print(f"--->>> 提示内容: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print("<<<--- LLM 调用结束！")
        # print(f"<<<--- 返回结果: {response}")
def callback_usage():
    # LangChain的Callback一般是由`BaseLLM`/`BaseChatModel`/`Chain`对象封装，不直接和Runnable基础类配合使用
    input_str = "请解释下机器学习算法SVM的原理"
    # 第1种方式
    client_llm_v1 = OpenAI(
        openai_api_key=API_KEY,
        openai_api_base=LLM_URL,
        model_name=MODEL,
        callbacks=[MyCustomHandler()]
    )
    res = client_llm_v1.invoke(input=input_str)
    print("--------------------------------")
    print(res)

    # 第2种方式
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

    # 第3种方式，在invoke方法里配置callback
    client_llm_v3 = OpenAI(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name=MODEL)
    # res = client_llm_v3.invoke(input=input_str, config={'callbacks': [MyCustomHandler()]})
    res = client_llm_v3.invoke(input=input_str, config={'callbacks': [MyCustomHandler()]})
    print("--------------------------------")
    print(res)


def memory_usage():
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


def chat_history_usage():
    # ----- 基于 BaseChatMessageHistory 实现的使用 -----
    # --- 单独使用 ---
    history = ChatMessageHistory()
    history.add_message(message=HumanMessage(content='hello from me'))
    history.add_message(message=AIMessage(content='hello from chat-llm'))
    print(history)
    print(history.messages)

    # --- 配合 Memory 组件使用 ---
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


def runnable_history_usage():
    # ----- 基于 RunnableWithMessageHistory 实现的使用 -----
    # RunnableWithMessageHistory 使用分为3个部分：

    # 1. 配置一个 Runnable 对象，Chain对象 或者 RunnableSequences对象 都可以
    # RunnableWithMessageHistory 主要是和 ChatModel + ChatPromptTemplate 配合使用的，
    # 它和 LLM + PromptTemplate 的搭配有问题：通过 history 插入的历史消息显示的是 HumanMessage/AIMessage 的字符串表示，而不是里面的 content。
    # client_llm = OpenAI(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name=MODEL)
    # template = """你是一个智能助手，负责回答用户的问题。对话历史:\n{history}\n用户输入:\n{user_input}\n请根据上下文生成回复："""
    # prompt = PromptTemplate(template=template, input_variables=["history", "user_input"])

    # 改为使用 ChatModel + ChatPromptTemplate
    client_chat = get_client_chat()
    prompt_chat = ChatPromptTemplate.from_messages(
        messages=[
            ("system", "你是一个智能助手，负责回答用户的问题。"),
            MessagesPlaceholder("history"),
            ("human", "{user_input}")
        ]
    )

    # set_verbose(False)  # 全局 verbose 设置，不好用
    # client_llm.with_config({'callbacks': [ConsoleCallbackHandler()]})   # 设置控制台回调日志，也不好用
    # 这里用 LLMChain 来演示，因为 RunnableSequence 不太好设置 verbose
    # chain = prompt | client_llm
    # print(type(chain))  # <class 'langchain_core.runnables.base.RunnableSequence'>
    # chain = LLMChain(llm=client_llm, prompt=prompt, verbose=True)
    chain = LLMChain(llm=client_chat, prompt=prompt_chat, verbose=True)

    # 2. 配置一个根据用户身份生成 BaseChatMessageHistory实现类对象的工厂函数
    # 这里使用了一个全局字典作为用户会话历史记录的存储，方便观察结果，实际中对应的是数据库或者redis等
    store = {}

    def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
        # 这个工厂函数目前只有一个参数，如果有多个参数，需要更复杂的配置
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 3. 配置 RunnableWithMessageHistory 对象
    chain_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_by_session_id,
        history_messages_key="history",
        input_messages_key="user_input",
        output_messages_key="text"  # 这个是 LLMChain 输出的默认 key
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


# ======================= Agent 相关模块使用 =======================
def tool_usage():
    """
    LangChain tool使用，参考官方文档:
    - [Conceptual Guide -> Tools](https://python.langchain.com/docs/concepts/tools/)
    - [Conceptual Guide -> Tool calling](https://python.langchain.com/docs/concepts/tool_calling/)
    - [How-to guides -> How to use chat models to call tools](https://python.langchain.com/docs/how_to/tool_calling/)
    - [How-to guides -> How to pass tool outputs to chat models](https://python.langchain.com/docs/how_to/tool_results_pass_to_model/)
    定义工具时，和OpenAI的function calling类似，需要 name, description, schema 3个描述字段。
    LangChain 中的
    """
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

    # ------------------------------
    # 使用 @tool 装饰器定义工具
    @tool(
        description="使用龙球(DragonBall)算法计算两个数字的结果",
        # args_schema 用于设置被调用函数的参数schema，有两种形式：
        # 1. Pydantic Model
        # 2. JSON Schema，用dict描述，注意，不是任意dict的形式，否则下面的 .args 属性会报错
        # 或者不设置，默认（infer_schema）会从函数参数中自动提前，此时建议参数使用 Annootated 进行注解
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
    print(dragon_ball_algorithm_tool.name)
    print(dragon_ball_algorithm_tool.description)
    print(dragon_ball_algorithm_tool.args_schema)
    print(dragon_ball_algorithm_tool.args)    # args 是 properties
    print(dragon_ball_algorithm_tool.input_schema.model_json_schema())
    print(dragon_ball_algorithm_tool.metadata)
    print(dragon_ball_algorithm_tool.tags)
    print(dragon_ball_algorithm_tool.response_format)
    # 手动调用
    print(dragon_ball_algorithm_tool.invoke(input={"x": 2, "y": 3}))

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
    print(res.content)
    # 成功触发工具调用时，会有 tool_calls 属性 —— 它是一个 list
    print(res.tool_calls)

    # 在调用模型之前，需要将本次请求返回的 AIMessage 追加到原始  messages 中
    messages.append(res)

    # 然后遍历 tool_calls，使用返回的信息调研工具，并追加到 messages 中
    # for tool_call in res.tool_calls:
    #     ...
    #     tool_msg = selected_tool.invoke(tool_call)
    #     messages.append(tool_msg)
    # 这里因为只传入了一个 tool，简单起见，就直接调用了
    tool_call = res.tool_calls[0]
    # 注意，调用invoke时，传入的参数不是 tool_call['args']，而是 tool_call -------------- KEY
    # 前者会返回值是内部tool的返回值，后者返回值被封装成了 ToolMessage，更方便使用一些
    # tool_call_res = dragon_ball_algorithm_tool.invoke(input=tool_call['args'])
    tool_call_res = dragon_ball_algorithm_tool.invoke(input=tool_call)
    print(type(tool_call_res))  # 确保这里返回的是 ToolMessage
    messages.append(tool_call_res)

    # 然后将 messages 传递给模型，并打印出结果
    res_final = client_chat_with_tool.invoke(input=messages)
    print(type(res_final))  # <class 'langchain_core.messages.ai.AIMessage'>
    print(res_final.content)

def tool_parser_usage():
    """
    上面 tool 的返回结果，可以使用 output_parser 里提供了如下 3 个工具类来直接解析函数调用的信息：
    - JsonOutputKeyToolsParser:  以 JSON 形式返回函数调用的参数
    - JsonOutputToolsParser:     以 JSON 形式返回函数调用中特定键的值
    - PydanticToolsParser:       将函数调用的参数作为 Pydantic 模型返回
    """
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
    messages.append( ToolMessage(content=res_tool_call_res, tool_call_id=res.tool_calls[0]['id']))
    final_res = client_chat_with_tool.invoke(input=messages)
    print(type(final_res))
    print(final_res.content)

def main():
    # llm_usage()
    # chat_llm_usage()
    # prompt_template_usage()
    # chat_prompt_template_usage()
    # placeholder_usage()
    # fewshot_prompt_template_usage()
    # pipeline_prompt_usage()
    simple_chat()
    # output_parser_usage()
    # structured_output_usage()
    # memory_usage()
    # chat_history_usage()
    # runnable_history_usage()
    # tool_usage()
    # tool_parser_usage()


if __name__ == '__main__':
    main()
