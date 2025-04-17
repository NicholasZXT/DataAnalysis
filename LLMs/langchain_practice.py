import os
# ----------
# from langchain import OpenAI
# --- LLM 模型包装器 ---
# from langchain.llms import OpenAI, ChatGLM, Tongyi  # 这个用法过时了，它只是从下面的 langchain_community.llms 中导入对应对象
# from langchain_community.llms import OpenAI, ChatGLM, Tongyi
# 上面的导入其实是从下面位置导入的包装器对象
from langchain_community.llms.openai import OpenAI
# 但是官方文档又提示OpenAI后续建议直接从下面的包里导入
# from langchain_openai.llms import OpenAI
from langchain_community.llms.chatglm import ChatGLM
from langchain_community.llms.tongyi import Tongyi
from langchain_community.llms.vllm import VLLM
# --- ChatLLM 模型包装器 ---
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatOpenAI, ChatHuggingFace, ChatLlamaCpp, ChatTongyi
# ChatOpenAI 官方文档建议直接从 langchain_openai 包中导入
# from langchain_openai.chat_models import ChatOpenAI
# ----------
from langchain_core.prompts import StringPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder, ChatMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, FewShotChatMessagePromptTemplate, PipelinePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import MarkdownListOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
# ----------
# document_loaders, embeddings, vectorstores, retrievers 都是从 langchain_community 包里的内容，官方建议直接从langchain_community包中导入
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, CSVLoader, JSONLoader, WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Cassandra, Clickhouse, Milvus, OpenSearchVectorSearch, \
    SKLearnVectorStore, ElasticsearchStore, ElasticVectorSearch, ElasticKnnSearch
from langchain_community.retrievers import BM25Retriever, ElasticSearchBM25Retriever
# ----------
from langchain_core.tracers.schemas import Run
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSequence, RunnableBinding
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager, StdOutCallbackHandler
# ----------
from langchain.chains.llm import LLMChain
from langchain_core.memory import BaseMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory, ChatMessageHistory, FileChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
# ----------
from langchain_core.tools import BaseTool, StructuredTool, tool
# from langchain.tools import ListDirectoryTool, ReadFileTool, WriteFileTool, HumanInputRun, ShellTool
from langchain_community.tools import ListDirectoryTool, ReadFileTool, WriteFileTool, HumanInputRun, ShellTool
# ----------
from langchain.globals import set_verbose
from langchain.callbacks.tracers import ConsoleCallbackHandler

API_KEY = 'Random'
LLM_URL = 'http://172.16.0.32:10086/v1'
# ======================= LLM + ChatLLM 模型包装器 使用 =======================
def llm_usage():
    client_llm = OpenAI(
        openai_api_key=API_KEY,
        openai_api_base=LLM_URL,
        model_name='Qwen2.5-32B-Instruct',
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        streaming=False,
        batch_size=20,
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
    client_chat = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=LLM_URL,
        model_name='Qwen2.5-32B-Instruct',
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        streaming=False,
        # batch_size=20,
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

    # Callable调用，不过只支持 List[BaseMessage] 参数
    msg = [SystemMessage(content='你是一个机器学习方面的专家'), HumanMessage(content='请问什么是SVM算法')]
    res = client_chat(messages=msg)
    print(res.content)


# ======================= PromptTemplate + Message 使用 =======================
def prompt_template_usage():
    # StringPromptTemplate含有抽象方法，不能实例化
    # pt = StringPromptTemplate(input_variables=["p1", "p2"], template="content-1: {p1}, content-2: {p2}")

    # PromptTemplate 使用
    template = "Tell me a {adjective} joke about {content}."
    # 第1种：直接实例化
    pt1 = PromptTemplate(input_variables=["adjective", "content"], template=template)
    pt1.format(adjective="funny", content="chickens")
    # 第2种：使用from_template方法——推荐这个
    pt2 = PromptTemplate.from_template(template=template)
    pt2.format(adjective="funny-2", content="chickens-2")
    print(pt2.template)
    print(pt2.template_format)
    print(pt2.input_variables)
    # 下面返回的是 <class 'langchain_core.prompt_values.StringPromptValue'>
    pv2 = pt2.format_prompt(adjective="funny-3", content="chickens-3")
    print(type(pv2))
    print(pv2)

def chat_prompt_template_usage():
    # ----- ChatMessagePromptTemplate 使用 -----
    template1 = "Tell me a {adjective} joke about {content}."
    # ChatMessagePromptTemplate 必须要指定 template 和 role
    cmpt = ChatMessagePromptTemplate.from_template(template=template1, role="user")
    cmpt_msg = cmpt.format(adjective="nice", content="fish")
    print(type(cmpt_msg))
    # <class 'langchain_core.messages.chat.ChatMessage'>
    print(cmpt_msg)
    print(cmpt_msg.content)
    print(cmpt_msg.type)  # chat
    print(cmpt_msg.role)  # user
    # 还有一个 format_messages 方法
    cmpt_msg_2 = cmpt.format_messages(adjective="nice", content="fish")
    # print(type(cmpt_msg_2))  # <class 'list'>
    print(type(cmpt_msg_2[0]))
    # <class 'langchain_core.messages.chat.ChatMessage'>

    template2 = "Tell me a {desc} joke about {something}."
    hmpt = HumanMessagePromptTemplate.from_template(template=template2)
    hmpt_msg = hmpt.format(desc="good", something="dog")
    print(type(hmpt_msg))
    # <class 'langchain_core.messages.human.HumanMessage'>
    print(hmpt_msg)
    print(hmpt_msg.content)
    print(hmpt_msg.type)   # human
    # HumanMessage 没有 role 属性！
    # print(hmpt_msg.role)

    # --- ChatPromptTemplate 用于组合多个ChatMessagePromptTemplate ---
    # 使用 from_messages() 方法，此方法接收一个 List
    # 其中的元素可以是：Union[BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate]
    # 使用 # List[BaseChatPromptTemplate] 创建时，后续的 format方法会起作用
    cpt = ChatPromptTemplate.from_messages(messages=[cmpt, hmpt])
    # 使用 # List[BaseChatPromptTemplate] 创建时，后续的 format方法就没啥用了
    cpt = ChatPromptTemplate.from_messages(messages=[cmpt_msg, hmpt_msg])  # List[BaseMessage]
    # --- format 方法 ---
    cpt_r1 = cpt.format(adjective="fantastic", content="cat", desc="laugh", something="rabbit")
    print(cpt_r1)
    # user: Tell me a nice joke about fish.
    # Human: Tell me a good joke about dog.
    # --- format_prompt 方法 ---
    cpt_r2 = cpt.format_prompt(adjective="fantastic", content="cat", desc="laugh", something="rabbit")
    print(type(cpt_r2))
    # <class 'langchain_core.prompt_values.ChatPromptValue'>
    print(cpt_r2)
    for msg in cpt_r2.messages:
        print(msg)
    # --- format_messages 方法 ---
    cpt_r3 = cpt.format_messages(adjective="fantastic", content="cat", desc="laugh", something="rabbit")
    print(type(cpt_r3))  # <class 'list'>
    print(type(cpt_r3[0]))
    # <class 'langchain_core.messages.chat.ChatMessage'>
    for msg in cpt_r3:
        print(msg)

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
    pass

# ======================= Output Parser 使用 =======================
# LangChain的输出解析器是和提示词配合使用的，它会在提示词的末尾增加一段要求大模型输出指定格式的指令。
def output_parser_usage():
    # 先实例化一个 parser 对象
    markdown_parser = MarkdownListOutputParser()
    # 可以查看该Parser的格式化提示词指令
    markdown_format_instructions = markdown_parser.get_format_instructions()
    # print(markdown_format_instructions)
    # 注意模版最后的 {format_instructions}，它并不在 input_variables 中填充
    template = "请解释下列机器学习算法的原理: {ml}\n{format_instructions}"
    prompt = PromptTemplate(
        input_variables=['ml'],
        template=template,
        # 设置 output_parser 参数，将 parser 对象传入到模板中
        output_parser=markdown_parser,
        # 通过 partial_variables 参数，将 parser 对象传入到模板中
        partial_variables={'format_instructions': markdown_format_instructions},
    )
    r1 = prompt.format(ml='SVM')
    print(r1)
    r2 = prompt.format_prompt(ml='SVM')
    print(r2.text)
    print(prompt.partial_variables)

    client_llm = OpenAI(
        openai_api_key=API_KEY,
        openai_api_base=LLM_URL,
        model_name='Qwen2.5-32B-Instruct',
        max_tokens=512,
    )
    res = client_llm.invoke(input=prompt.format_prompt(ml='SVM'))
    print(res)

    # 调用模型之后，使用如下方式解析模型输出
    res_parse = markdown_parser.parse(text=res)
    for line in res_parse:
        print(line)

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


def chain_usage():
    client_llm = OpenAI(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name='Qwen2.5-32B-Instruct')
    client_chat = ChatOpenAI(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name='Qwen2.5-32B-Instruct')
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
        model_name='Qwen2.5-32B-Instruct',
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
        model_name='Qwen2.5-32B-Instruct',
        callbacks=callback_manager,
        # 或者下面这个
        # callback_manager=callback_manager,
    )
    res = client_llm_v2.invoke(input=input_str)
    print("--------------------------------")
    print(res)

    # 第3种方式，在invoke方法里配置callback
    client_llm_v3 = OpenAI(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name='Qwen2.5-32B-Instruct')
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
    client_llm = OpenAI(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name='Qwen2.5-32B-Instruct')
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
    client_llm = OpenAI(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name='Qwen2.5-32B-Instruct')
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
    template = "对话历史:\n{history}\n用户输入: {input}\n请你回复."
    prompt = PromptTemplate(template=template, input_variables=["history", "input"])
    # template = "对话历史:\n\n用户输入: {input}\n请你回复."
    # prompt = PromptTemplate(template=template, input_variables=["input"])
    client_llm = OpenAI(openai_api_key=API_KEY, openai_api_base=LLM_URL, model_name='Qwen2.5-32B-Instruct')
    # set_verbose(False)  # 全局 verbose 设置，不好用
    # client_llm.with_config({'callbacks': [ConsoleCallbackHandler()]})   # 设置控制台回调日志，也不好用
    # 这里用 LLMChain 来演示，因为 RunnableSequence 不太好设置 verbose
    # chain = prompt | client_llm
    # print(type(chain))  # <class 'langchain_core.runnables.base.RunnableSequence'>
    chain = LLMChain(llm=client_llm, prompt=prompt, verbose=True)

    # 2. 配置一个根据用户身份生成 BaseChatMessageHistory实现类对象的工厂函数
    # 这里使用了一个全局字典作为用户会话历史记录的存储，方便观察结果，实际中对应的是数据库或者redis等
    store = {}
    # 这个工厂函数目前只有一个参数，如果有多个参数，需要更复杂的配置
    def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 3. 配置 RunnableWithMessageHistory 对象
    chain_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_by_session_id,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 4. 调用 RunnableWithMessageHistory 对象的 invoke 方法，用户身份通过 config 参数设置
    # >>> 用户1的会话
    print("------------ user-1 -----------------")
    u1_r1 = chain_with_history.invoke(input={"input": "你好，我先和你打个招呼"}, config={"configurable": {"session_id": "user-1"}})
    print(u1_r1)
    # print(store)
    # print(store.keys())
    print(">>>>> user-1 chat-2")
    u1_r2 = chain_with_history.invoke(input={"input": "我们刚才聊了什么"}, config={"configurable": {"session_id": "user-1"}})
    print(u1_r2)
    # print(store.keys())

    # >>> 用户2的会话
    print("------------ user-2 -----------------")
    u2_r1 = chain_with_history.invoke(input={"input": "你好，我想和你聊聊历史"}, config={"configurable": {"session_id": "user-2"}})
    print(u2_r1)
    print(store.keys())

    u2_r2 = chain_with_history.invoke(input={"input": "我们刚才聊了什么"}, config={"configurable": {"session_id": "user-2"}})
    print(u2_r2)
    print(store.keys())


# ======================= Agent 相关模块使用 =======================
@tool
def multiply_tool(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def multiply_func(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def tool_usage():
    # 只有 部分 ChatLLM 支持 bind_tools
    # client_chat = ChatGLM(
    client_chat = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=LLM_URL,
        model_name='Qwen2.5-32B-Instruct',
        max_tokens=512,
    )
    # client_chat = ChatTongyi(
    #     dashscope_api_key=API_KEY,
    #     openai_api_base=LLM_URL,
    #     model_name='Qwen2.5-32B-Instruct',
    #     max_tokens=512,
    # )
    # client_chat.bind()  # 这个方法是绑定运行时参数
    # client_chat.bind_functions()   # 这个方法只有 ChatOpenAI 有，对应OpenAI的 function_call 功能配置
    # client_chat.bind_tools()  # bind_tools 方法也不是所有 ChatLLM 都有的，比如 ChatGLM 就没有，但是 ChatTongyi 有

    # 检查下tool的封装
    print(type(multiply_tool))
    # <class 'langchain_core.tools.structured.StructuredTool'>
    print(multiply_tool.name)
    print(multiply_tool.description)
    print(multiply_tool.args)
    print(multiply_tool.args_schema)
    print(multiply_tool.metadata)
    print(multiply_tool.tags)
    print(multiply_tool.response_format)
    # 手动调用
    print(multiply_tool.invoke({"a": 2, "b": 3}))

    # Langchain-community 提供的现成工具
    ls_tool = ListDirectoryTool()
    print(type(ls_tool))
    # <class 'langchain_community.tools.file_management.list_dir.ListDirectoryTool'>
    print(ls_tool.name)
    print(ls_tool.description)
    print(ls_tool.args)
    print(ls_tool.args_schema)
    res = ls_tool.invoke(input={'dir_path': './LLMs'})
    print(res)

    # ChatOpenAI 只支持 bind_functions，不支持 bind_tools
    # client_chat_with_tool = client_chat.bind_tools(tools=[multiply_tool])
    # client_chat_with_tool = client_chat.bind_functions(functions=[multiply], function_call='auto')
    client_chat_with_tool = client_chat.bind_functions(functions=[multiply_func])
    messages = [
        {'role': 'system', 'content': '你是一个算术专家'},
        {'role': 'user', 'content': '请计算一下1024乘以1024的结果'},
    ]
    res = client_chat_with_tool.invoke(input=messages)
    print(res)
    # print(res.content)


def main():
    # memory_usage()
    # chat_history_usage()
    runnable_history_usage()


if __name__ == '__main__':
    main()
