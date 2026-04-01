"""
简单研究 LangGraph 的使用 —— 适用于 v0.6.x / v1.x 版本。

主要参考了如下官方文档：
- [LangGraph Glossary](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

首先需要明确的是，LangGraph 不依赖 Langchain-Core 或者 Langchain，因此下面的研究都使用一个简单的Python Callable 对象
来代替实际中的 Langchain-Core/Langchain 里的 Runnable/LLM/ChatModel/Chain 对象。

即使升级到 LangGraph v1.0.x 版本，LangGraph 的核心架构和组件都没有太大的变化。
"""
# %%
from typing import Annotated, List, TypedDict, Dict, Union, Iterator
from dataclasses import dataclass
# from typing_extensions import TypedDict
# ---------- LangGraph Graph 组件 ----------
from langgraph.constants import START, END
# from langgraph.graph.graph import Graph, CompiledGraph  # 这两个类只有 v0.4.10 版本之前有，v0.5.0开始被删除了
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import MessageGraph, MessagesState, add_messages
# --- Functional API ---
from langgraph.func import entrypoint, task
# ---------- LangGraph Memory/Store 组件 ----------
from langgraph.checkpoint.base import CheckpointTuple, Checkpoint, CheckpointMetadata, BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store
from langgraph.types import StateSnapshot
# ---------- LangGraph HIL 工具 ----------
from langgraph.types import interrupt, Command, Send
# ---------- LangGraph Tool 组件 ----------
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent, InjectedState, InjectedStore
from langgraph.runtime import Runtime
from langchain.tools import ToolRuntime  # LangChain 里也提供了一个类似的 ToolRuntime 类
from langgraph.utils.runnable import RunnableCallable
# from langgraph.prebuilt.chat_agent_executor import AgentState
# ---------- langchain-core 组件 ----------
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import BaseTool, StructuredTool, tool
# ---------- 其他依赖 ----------
import json

# --- vLLM 部署 ---
# API_KEY = 'Empty'
# LLM_URL = 'http://172.16.0.32:10086/v1'
# MODEL = 'Qwen2.5-32B-Instruct'
# --- Ollama 本地部署 ---
API_KEY = 'Empty'
LLM_URL = 'http://localhost:11434'
MODEL = 'qwen2.5:7b'
# MODEL = 'qwen3:8b'

# %%
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


# %% ======================= 无状态图 构建 =======================
def stateless_graph_usage():
    """
    Graph / CompiledGraph 只有 v0.4.10 版本之前有，v0.5.0 版本开始删除了这两个类所在的 langgraph.graph.graph.py 文件.
    因此这两个类不需要关注了。
    """
    ...
    # 创建一个 Graph，这个Graph类不接受任何初始化参数，所以说它是无状态的。
    # graph = Graph()
    # 定义节点
    # graph.add_node(node="hello", action=lambda _: "Hello, world !")
    # graph.add_node(node="welcome", action=lambda _: "Welcome LangGraph !")
    # graph.add_node(node="bye", action=lambda _: "Goodbye !")

    # 定义边
    # graph.add_edge("hello", "welcome")
    # graph.add_edge("welcome", "bye")

    # 定义起始点和结束点
    # graph.set_entry_point('hello')
    # graph.set_finish_point('bye')

    # 编译并执行
    # compile_graph: CompiledGraph = graph.compile(name='StatelessGraph')
    # print(compile_graph.config_specs)
    # print(compile_graph.config_type)
    # result = compile_graph.invoke(input={'key1': 'value1', 'key2': 'value2'})
    # print(result)


# %% ======================= 简单有状态图 构建 =======================
def stateful_graph_usage():
    """
    LangGraph 的 StateGraph 基本使用
    """
    # 1. 首先定义整个 Graph 的状态表示，可以直接用dict，也可以用 TypedDict，或者是 Pydantic的 BaseModel —— 状态表示完全由用户自定义
    class SimpleState(TypedDict):
        """
        这个 state 对象的 key 就是后续 invoke() 方法里 input 参数接受的 dict 的key
        """
        messages: List[str]
        count: int

    # 2. 定义Node里要运行的Python函数
    def greet_node(state: SimpleState) -> SimpleState:
        print(f"--> greet_node start...")
        print(f"  state: {state}")
        state["messages"].append("Hello")
        print(f"<-- greet_node end.")
        return state

    def increment_node(state: SimpleState) -> SimpleState:
        print(f"--> increment_node start...")
        print(f"  state: {state}")
        state["count"] += 1
        print(f"<-- increment_node end.")
        return state

    def something_node(state: SimpleState) -> SimpleState:
        print(f"--> something_node start...")
        print(f"  state: {state}")
        state["messages"].append("Something")
        state["count"] += 2
        print(f"<-- something_node end.")
        return state

    def partial_node(state: SimpleState) -> Dict[str, int]:
        print(f"--> partial_node start...")
        print(f"  state: {state}")
        print(f"<-- partial_node end.")
        # 只返回状态部分的key也可以，不过此时如果对应的 key 没有设置 reducer 函数的话，默认会覆盖该 key 的内容
        return {'count': 10}

    # 3. 使用StateGraph 构建 Graph，初始化参数必须传入自定义的 State 类
    # state_schema 指定的 State 类，对应的就是 invoke() 方法里 input= 参数的 dict 结构 -------------------- KEY
    graph = StateGraph(state_schema=SimpleState)

    # 4. 使用 add_node 方法添加节点，add_node 方法有多个重载，注意选择
    graph.add_node(node="greet_node", action=greet_node)
    graph.add_node(node="increment_node", action=increment_node)
    graph.add_node(node="something_node", action=something_node)
    graph.add_node(node="partial_node", action=partial_node)

    # 5. 定义边
    graph.set_entry_point("greet_node")
    graph.add_edge("greet_node", "increment_node")
    graph.add_edge("increment_node", "something_node")
    graph.add_edge("something_node", "partial_node")
    graph.set_finish_point("partial_node")

    # 6. 编译构建 Graph，返回的是 CompiledStateGraph 对象
    compile_graph: CompiledStateGraph = graph.compile(name='SimpleStateGraph')
    # 可以查看具体的图结构
    # graph_picture = compile_graph.get_graph()
    print(compile_graph.config_specs)
    print(compile_graph.name)
    # print(compile_graph.get_name())

    # 7. 使用 Graph
    # 运行Graph: 调用 invoke/ainvoke; stream/astream 方法
    # invoke()方法的 input= 参数接受一个 TypedDict/dataclass/pydantic BaseModel，对应于初始化时 state_schema 定义的结构 ------ KEY
    input = {"messages": [], "count": 0}
    print(f"Graph input: {input}")
    res = compile_graph.invoke(input=input)
    print(type(res))  # 早期版本是 <class 'langgraph.pregel.io.AddableValuesDict'>，新版本是 <class 'dict'>
    # 早期 Graph 返回的 AddableValuesDict 只是一个对 dict 进行简单封装，重写了 __add__ 方法的dict, key 和 state 里定义的完全一样
    print(f"res.keys: {res.keys()}")  # dict_keys(['messages', 'count'])，key 和 state 里定义的完全一样
    print(f"output state: {res}")

    # 批量调用: batch/abatch 方法
    inputs = [{"messages": [], "count": 0}, {"messages": ["Hi"], "count": 1}]
    res_batch = compile_graph.batch(inputs=inputs)
    # print(f"{type(res_batch)}, {len(res_batch)}, {type(res_batch[0])}")
    # <class 'list'>, 2, <class 'langgraph.pregel.io.AddableValuesDict'>
    # batch 调用时，返回的是 List[AddableValuesDict]
    for item in res_batch:
        print(f"final state: {item}")


# %%
def message_graph_usage():
    """
    对于LLM场景，LangGraph 还提供了 MessageState 和 MessageGraph，方便直接使用。
    - MessageGraph 也有一个存放了 List[BaseMessage]的状态，但是似乎是匿名的，不能指定key。
    - MessagesState 就是一个 TypedDict，含有一个 messages 的key，存放了 List[BaseMessage]，并使用Annotated标注使用了 add_message()

     add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]
     整体的核⼼逻辑是合并两个消息列表，按 ID 更新现有消息。
     默认情况下，状态为“仅附加”，当新消息与现有消息具有相同的 ID时，进⾏更新。
     合并逻辑则是：如果right的消息与left的消息具有相同的 ID，则right的消息将替换left的消息，否则作为⼀条新的消息进⾏追加。
     返回值是合并后的 List[BaseMessage]。
    """
    def some_node(state: MessagesState):
        print("--> some_node start...")
        print(f"  state: {state}")
        print("<-- some_node end...")
        # 不能返回 dict，只能返回 List[BaseMessage]
        # return {'messages': [HumanMessage(content="some-node")]}
        return [HumanMessage(content="some-node")]

    graph = MessageGraph()
    graph.add_node(node="some_node", action=some_node)
    graph.set_entry_point("some_node")
    graph.set_finish_point("some_node")
    compile_graph = graph.compile(name='MessageGraph')

    # 同样，input_msg 也不能是 dict，而是 List[BaseMessage]
    # input_msg = {"messages": [HumanMessage(content="Hello World!")]}
    input_msg = [HumanMessage(content="Hello World!")]
    res = compile_graph.invoke(input=input_msg)
    print(res)


# %% ======================= 基于条件动态执行有状态图 =======================
def graph_conditional_usage():
    """
    展示有条件动态执行的状态图.
    主要是 Graph.add_conditional_edges() 方法的使用，该方法参数如下：
    - source: str, 指定 source 节点
    - path: 一个Python Callable 对象
      - 接受的参数是当前图的状态state
      - 在没有下面的 path_map 情况下，返回值会被作为 下一个/多个 执行Node的名称，多个Node会被并行执行
      - 如果配置的 path_map，则返回值会被作为 path_map 的key，映射到具体的下一个Node名称
    - path_map: dict[Hashable, str] | list[str]，对 path 的返回值进行映射处理
      - 以 dict 形式给出时，会将 path 的返回值作为 key，获取实际待执行的下一个 Node 名称
      - 以 list[str] 形式给出时，则表示后续执行的Node名称。实际上在 BranchSpec.from_path() 方法里会被转换成 {item: item} 的恒等映射dict
    """
    class SimpleState(TypedDict):
        messages: List[str]
        count: int

    def greet_node(state: SimpleState) -> SimpleState:
        state["messages"].append("Hello")
        state["count"] += 1
        print(f"--> greet_node running...")
        return state

    def reset_count_node(state: SimpleState):
        state["messages"].append("Reset Count")
        state["count"] = 0
        print(f"--> reset_count_node running...")
        return state

    def conditional_edge_check(state: SimpleState):
        print(f"==> conditional_edge_check running...")
        if state["count"] > 3:
            print(f"  Switch to -> reset_count_node")
            return "reset_count_node"
        else:
            print(f"  Switch to -> END")
            return END

    graph = StateGraph(state_schema=SimpleState)
    graph.add_node(node="greet_node", action=greet_node)
    graph.add_node(node="reset_count_node", action=reset_count_node)

    graph.set_entry_point("greet_node")
    # 根据state当前值选择下一个执行Node
    graph.add_conditional_edges(source="greet_node", path=conditional_edge_check)

    compile_graph: CompiledStateGraph = graph.compile(name='StateGraphWithConditionalEdges')

    res1 = compile_graph.invoke(input={"messages": [], "count": 0})
    print(res1)
    print("---------------------------------------")
    res2 = compile_graph.invoke(input={"messages": [], "count": 3})
    print(res2)


# %% ======================= Graph Checkpoint（短期记忆） 使用 =======================
def graph_checkpoint_usage():
    """
    checkpoint 机制是 LangGraph 提供的短期记忆机制。

    这里的短期记忆机制有两点含义：
    - 区分不同会话的历史对话记录，这个通过 LangGraph引入的 thread_id 来区分不同的会话。
    - 保存当前会话的历史对话记录（不仅是本次对话历史，还有之前的对话历史）

    为了展示 checkpoint 的效果，定义的 state 对象里，每个属性需要有一个 reducer 函数，这里使用了两种：
    1. 自定义 reducer
    2. LangGraph提供的 add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage] 函数
    reducer 函数是为了保存/合并 本次对话历史，但它不能保存之前的对话历史。
    """
    def num_reducer(prev_num: List[int], curr_num: List[int]) -> List[int]:
        return prev_num + curr_num

    class ReduceState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]
        num: Annotated[List[int], num_reducer]

    def greet_node(state: ReduceState) -> Dict[str, List[str]]:
        """
        此Node负责更新 ReduceState 里的 messages。
        """
        print(f"--> greet_node start...")
        print(f"  state: {state}")
        if len(state['messages']) > 0:
            # 获取状态里 messages 列表的最后一条消息（如果有）的内容
            # state_msg = state['messages'][-1]
            state_msg = state['messages'][-1].content
        else:
            # message没有消息，则默认为 Hello
            state_msg = 'Hello'
        if len(state["num"]) > 0:
            # 获取状态里的 num 列表的最后一个数字（如果有）
            state_num = state['num'][-1] + 1
        else:
            # num 列表为空，则默认为 0
            state_num = 0
        # 基于状态里最后信息，生成 message 列表里的新消息
        new_message = f"{state_msg}[{state_num}]"
        print(f"<-- greet_node end.")
        return {'messages': [new_message]}

    def increment_node(state: ReduceState) -> Dict[str, List[int]]:
        """
        此Node负责更新 ReduceState 里的 num。
        """
        print(f"--> increment_node start...")
        print(f"  state: {state}")
        if len(state["num"]) > 0:
            # 获取状态里 num 列表的最后一个数字（如果有），并递增
            state_num = state['num'][-1] + 1
        else:
            # state 里 num 列表为空，则默认为 0
            state_num = 0
        print(f"<-- increment_node end.")
        return {'num': [state_num]}

    graph = StateGraph(state_schema=ReduceState)
    graph.add_node(node="greet_node", action=greet_node)
    graph.add_node(node="increment_node", action=increment_node)
    graph.set_entry_point("greet_node")
    graph.add_edge("greet_node", "increment_node")
    graph.set_finish_point("increment_node")

    # 这个就是 checkpoint 对象，在compile的时候传入
    memory = MemorySaver()  # 基于内存的checkpoint简单实现
    compile_graph: CompiledStateGraph = graph.compile(name='StateGraphWithCheckpoint', checkpointer=memory)

    print("-------- user-1 call-1 ------------")
    # 调用的时候传入一个config字段，key 必须是 configurable，里面设置一个 thread_id，用于表示当前用户身份
    u1_config = {"configurable": {"thread_id": "user-1"}}

    u1_input_1 = {"messages": [], "num": []}
    print(f"u1_input_1: {u1_input_1}")
    u1_r1 = compile_graph.invoke(input=u1_input_1, config=u1_config)
    # print(type(u1_r1))  # <class 'langgraph.pregel.io.AddableValuesDict'>
    # print(f"u1_r1 state: {u1_r1}")
    print(f"==> u1_r1 state:")
    for msg, num in zip(u1_r1["messages"], u1_r1["num"]):
        msg.pretty_print()
        print(f"num: {num}")

    # 获取当前的状态，必须要使用 invoke 时同样的 config
    u1_r1_state: StateSnapshot = compile_graph.get_state(config=u1_config)
    # print(type(u1_r1_state))  # <class 'langgraph.types.StateSnapshot'>
    # print(u1_r1_state)
    print("\n==> u1_r1_state show:")
    print('  u1_r1_state.config: ', u1_r1_state.config)
    print('  u1_r1_state.metadata: ', u1_r1_state.metadata)
    # 下面就是当前 state 对象的值，应该和 u1_r1 的内容是一样的
    print('  type(u1_r1_state.values): ', type(u1_r1_state.values))  # <class 'dict'>
    print('  u1_r1_state.values:\n', u1_r1_state.values)

    print("\n-------- user-1 call-2 ------------")
    u1_input_2 = {"messages": ["Call-2"], "num": [10]}
    print(f"u1_input_2: {u1_input_2}")
    u1_r2 = compile_graph.invoke(input=u1_input_2, config=u1_config)
    # print(f"u1_r2 state: {u1_r2}")
    print("==> u1_r2 state:")
    for msg, num in zip(u1_r2["messages"], u1_r2["num"]):
        msg.pretty_print()
        print(f"num: {num}")
    u1_r2_state = compile_graph.get_state(config=u1_config)
    print("\n==> u1_r2_state show:")
    # print('  u1_r2_state.config: ', u1_r2_state.config)
    # print('  u1_r2_state.metadata: ', u1_r2_state.metadata)
    print('  u1_r2_state.values:\n', u1_r2_state.values)

    # ---- 获取历史状态，这也是 TimeTravel 的原理，获取历史状态，然后Replay ----
    print("\n-------- History State ------------")
    history_states: Iterator[StateSnapshot] = compile_graph.get_state_history(u1_config)
    for state in history_states:
        # print(type(state))  # <class 'langgraph.types.StateSnapshot'>
        # print(state)
        print("==> state show:")
        print('  state.metadata: ', state.metadata)  # 这个有用
        print('  state.values: ', state.values)  # 这个有用
        # print('  state.config: ', state.config)
        # print('  state.tasks: ', state.tasks)
        # print('  state.next: ', state.next)
        # print('  state.parent_config: ', state.parent_config)

    # ---- 从 Memory 对象里获取所有 checkpoint 列表 ----
    print("\n-------- Checkpoints ------------")
    checkpoint_iter: Iterator[CheckpointTuple] = memory.list(config=u1_config)
    # 下面展示的checkpoint顺序是倒序的 ---------- KEY
    for checkpoint in checkpoint_iter:
        # print(type(checkpoint))   # <class 'langgraph.checkpoint.base.CheckpointTuple'>
        # print(checkpoint)
        print("==> checkpoint show:")
        print('  checkpoint.config: ', checkpoint.config)
        print('  checkpoint.metadata: ', checkpoint.metadata)  # 这个信息最有用
        # print('  checkpoint.pending_writes: ', checkpoint.pending_writes)
        # print('  checkpoint.checkpoint: ', checkpoint.checkpoint)
        # print('  checkpoint.parent_config: ', checkpoint.parent_config)


# %% ======================= Graph Store（长期记忆） 使用 =======================
def graph_store_usage():
    """
    LangGraph 的长期记忆是用于跨用户（thread_id）存储的。
    LangGraph的长期记忆存储的抽象基类是 langgraph.store.base.BaseStore。
    它采用 namespace + key + value 的层次来组织存储：
    - namespace: 支持多层次命名空间，使用 Tuple[str,...]来表示，比如 (user_id, application_name)
    - key: 对应命名空间下存储的 key
    - value: JSON结构，以dict形式存储
    """
    memory_store = InMemoryStore()
    memory_store.put(
        namespace=("user-1", "web"),
        key="web-k-1",
        value={"some": "some-value"}
    )
    memory_store.put(
        namespace=("user-1", "db"),
        key="db-k-1",
        value={"some": "some-value"}
    )
    print(memory_store.list_namespaces())
    print(memory_store.get(namespace=("user-1", "web"),  key="web-k-1"))
    print("----------------------------")

    def some_node(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Dict[str, List[BaseMessage]]:
        """
        此Node中通过 RunnableConfig 类型拿到了运行时的配置，BaseStore 类型提示拿到了 store 配置。
        此外，langgraph 还提供了 get_store() 函数用于获取 store 配置。
        """
        print("--> some_node start...")
        print(f"  state: {state}")
        # 可以从 config 对象中获取 invoke 里传入的 config 参数信息，比如其中的 user_id 的内容，之后再查询用户相关的信息
        # print(f"  config: {config}")
        print(f"  use_id: {config.get('configurable', {}).get('user_id', '')}")
        # 在节点内部，可以通过 store = get_store() 来获取 store 对象  ----------- KEY
        store = get_store()
        store_namespaces = []
        for item in store.list_namespaces():
            store_namespaces.append('.'.join(item))
        store_namespaces = ';'.join(store_namespaces)
        print(f"  store-namespaces: {store_namespaces}")
        print("<-- some_node end...")
        return {'messages': [HumanMessage(content=store_namespaces)]}

    graph = StateGraph(MessagesState)
    graph.add_node(node="some_node", action=some_node)
    graph.set_entry_point("some_node")
    graph.set_finish_point("some_node")
    compile_graph = graph.compile(name='GraphWithStore', store=memory_store)

    # input_msg = {"messages": [{"role": "user", "content": "请列出当前Store的namespace"}]}
    input_msg = {"messages": [HumanMessage(content="请列出当前Store的namespace")]}
    config = {"configurable": {"user_id": "user-1"}}
    res = compile_graph.invoke(input=input_msg, config=config)
    # print(res)
    for msg in res['messages']:
        msg.pretty_print()


# %% ======================= Interrupt/Command (HIL) 机制 =======================
# 个人感觉 LangGraph 的HIL机制设计的不是很好用。
# 因为HIL触发时，会从 graph.invoke() 返回，此时需要用户在这里对返回的内容进行判断是否包含 Interrupt 信息：
# - 如果包含，则进行 HIL 干预后，再次调用 graph.invoke() 恢复执行
# - 如果不包含，那么拿到的就是本次输入的最终结果
# 也就是说，本来是单次调用 invoke() 的流程，现在不得不使用一个 while 循环来处理 invoke() 的 interrupt 流程；
# 更麻烦的是，如果有多个节点会触发 HIL，那还需要在这个 while 循环里增加一个 if 或者 switch 判断，处理多个 HIL 的情况。
def graph_dynamic_interrupt_usage():
    """
    展示 LangGraph 的 Human-Interrupt 使用 —— 动态断点设置。
    通过 interrupt() 函数 + Command 对象 实现断点触发及恢复。

    这种方式可以在 node 内执行一些断点处理逻辑。
    实际上，查看 interrupt() 的源码可以发现，它其实是抛出一个 GraphInterrupt 异常来实现中断Graph执行的。

    注意，使用 interrupt() 在 node 内动态设置断点，后续使用 Command 对象恢复执行时，
    **整个 node 内的代码都会被重新执行一次！！！**
    而不是从 interrupt() 函数的部分恢复执行（类似 yield 的效果）。
    """
    def num_reducer(prev_num: List[int], curr_num: List[int]) -> List[int]:
        return prev_num + curr_num

    class HumanInterruptState(TypedDict):
        num: Annotated[List[int], num_reducer]
        human_msg: str

    def greet_node(state: HumanInterruptState) -> Dict[str, str]:
        print(f"--> greet_node start...")
        if len(state["num"]) > 0:
            value = {"state.num": state['num']}
        else:
            value = {"state.num": 0}
        # --------------------------------------------------------------------------
        print(f"  ==> greet_node is waiting for human response with value: {value}")
        # 使用 interrupt 函数打断图的执行，等待人工输入 --------------- KEY
        # interrupt() 函数的参数 value 会被返回
        human_response = interrupt(value=value)
        # interrupt() 函数的返回值就是断点恢复执行时通过 Command 对象的 resume 参数设置的值
        print(f"  <== greet_node received human interrupt response: {human_response}")
        # 但是有一点需要特别注意：断点恢复执行时，整个 greet_node 里的逻辑都会被重新执行，
        # 而不是从 interrupt() 函数返回后的部分继续执行（类似于 yield 的效果）
        # --------------------------------------------------------------------------
        print(f"<-- greet_node end.")
        return {'human_msg': human_response["human_msg"]}

    def increment_node(state: HumanInterruptState) -> Dict[str, List[int]]:
        print(f"--> increment_node start...")
        if len(state["num"]) > 0:
            state_num = state['num'][-1] + 1
        else:
            state_num = 0
        print(f"<-- increment_node end.")
        return {'num': [state_num]}

    graph = StateGraph(state_schema=HumanInterruptState)
    graph.add_node(node="greet_node", action=greet_node)
    graph.add_node(node="increment_node", action=increment_node)
    graph.set_entry_point("greet_node")
    graph.add_edge("greet_node", "increment_node")
    graph.set_finish_point("increment_node")

    # HIL 需要借助 checkpointer 才能使用
    memory = MemorySaver()
    compile_graph: CompiledStateGraph = graph.compile(name='StateGraphWithDynamicHIL', checkpointer=memory)

    u1_config = {"configurable": {"thread_id": "user-1"}}
    u1_input = {"messages": '', "num": []}
    print(f"u1_input = {u1_input}")
    u1_r1 = compile_graph.invoke(input=u1_input, config=u1_config)
    print(f"u1_r1 state: {u1_r1}")
    # 通过 interrupt() 函数触发 Human-Interrupt 时，value 参数可以通过 __interrupt__ 这个key获取
    print(f"__interrupt__: {u1_r1['__interrupt__']}")
    print("--------------------")
    resume = {'human_msg': "hello world"}
    print(f"resume: {resume}")
    # 断点恢复时，通过 Command 对象传入的 resume 参数的值，会被作为 interrupt() 函数的返回值
    u1_r1_command = Command(resume=resume)
    u1_r1_continue = compile_graph.invoke(input=u1_r1_command, config=u1_config)
    print(f"u1_r1_continue: {u1_r1_continue}")


# %%
def graph_fixed_interrupt_usage():
    """
    展示 LangGraph 的 Human-Interrupt 使用 —— 固定断点设置。
    在 Graph.compile() 方法里通过 interrupt_before 参数，指定在某些 node 前/后 设置断点。
    这种方式把断点时的处理逻辑放到了 Graph 外面。
    """
    class HumanInterruptState(TypedDict):
        msg: Annotated[BaseMessage, add_messages]

    def greet_node(state: HumanInterruptState) -> Dict[str, str]:
        print(f"--> greet_node start...")
        print(f"  state: {state}")
        print(f"<-- greet_node end.")
        return {'msg': "Hello LangGraph"}

    def show_node(state: HumanInterruptState) -> Dict[str, str]:
        print(f"--> show_node start...")
        print(f"  state: {state}")
        print(f"<-- show_node end.")
        return {'msg': "I'm LangGraph"}

    graph = StateGraph(state_schema=HumanInterruptState)
    graph.add_node(node="greet_node", action=greet_node)
    graph.add_node(node="show_node", action=show_node)
    graph.set_entry_point("greet_node")
    graph.add_edge("greet_node", "show_node")
    graph.set_finish_point("show_node")

    # HIL 需要借助 checkpointer 才能使用
    memory = MemorySaver()
    compile_graph: CompiledStateGraph = graph.compile(
        name='StateGraphWithFixedHIL', checkpointer=memory,
        interrupt_before=["show_node"],  # 指定在某些节点前设置断点
        # interrupt_after=["greet_node"]    # 指定在某些节点后设置断点
    )

    u1_config = {"configurable": {"thread_id": "user-1"}}
    u1_input = {"msg": []}
    print(f"u1_input = {u1_input}")
    u1_r1 = compile_graph.invoke(input=u1_input, config=u1_config)
    # 控制台日志可以看到，只执行了 greet_node 节点，show_node 节点没有执行
    print(type(u1_r1))  # <class 'dict'>
    print(f"u1_r1 state: {u1_r1}")
    # {'msg': [HumanMessage(content='Hello LangGraph', additional_kwargs={}, response_metadata={}, id='7685ea03-46b0-467c-9c5c-eab8ad627935')]}
    # -------------------------
    # 在断点停住之后，可以使用 graph.get_state() 方法获取当前的状态；使用 graph.update_state() 方法来修改状态
    u1_r1_state: StateSnapshot = compile_graph.get_state(config=u1_config)
    print(u1_r1_state.values)
    # 修改状态
    conf: RunnableConfig = compile_graph.update_state(config=u1_config, values={'msg': 'Hello LangGraph from HIL'})
    # u1_r1_state.values['msg'] = [HumanMessage(content='Hello LangGraph from HIL')]
    # compile_graph.update_state(config=u1_config, values=u1_r1_state)
    print(conf)
    # 检查修改的状态
    s = compile_graph.get_state(config=u1_config)
    print(s.values)
    # -------------------------
    # 恢复执行，input输入 None，这里恢复时，会从上次中断点继续执行，即从 show_node 节点开始执行，而不会再次执行 greet_node 节点
    u1_r1_continue = compile_graph.invoke(input=None, config=u1_config)
    print(f"u1_r1_continue: {u1_r1_continue}")


# %% ======================= 结合 LangChain 的 ChatBot 案例 =======================
def chatbot_example():
    class MsgState(TypedDict):
        messages: Annotated[list[Union[str, BaseMessage]], add_messages]

    graph = StateGraph(MsgState)

    client_chat = get_client_chat()
    # res = client_chat.invoke(input=[{'role': 'user', 'content': '你好，可以和我聊聊历史吗？'}])
    # print(res.content)
    def chatbot_node(state: MsgState):
        print(f"--> chatbot_node start...")
        print(f"  state: {state}")
        response = client_chat.invoke(input=state["messages"])
        print(f"<-- chatbot_node end.")
        return {"messages": [response]}

    graph.add_node(node='chatbot', action=chatbot_node)
    graph.set_entry_point('chatbot')
    graph.set_finish_point('chatbot')

    memory = MemorySaver()
    compile_graph = graph.compile(name='ChatBotGraph', checkpointer=memory)

    u1_config = {"configurable": {"thread_id": "user-1"}}
    print("-------- user-1 chat-round-1 ------------")
    msg_1 = [{'role': 'user', 'content': '你好，可以和我聊聊历史吗？'}]
    u1_r1 = compile_graph.invoke(input={"messages": msg_1}, config=u1_config)
    # print(type(u1_r1))  # <class 'langgraph.pregel.io.AddableValuesDict'>
    # print(u1_r1['messages'])
    for msg in u1_r1['messages']:
        # print(msg.content)
        msg.pretty_print()
    u1_r1_state = compile_graph.get_state(config=u1_config)
    # print("\n---> u1_r1_state show:")
    # print('  u1_r1_state.config: ', u1_r1_state.config)
    # print('  u1_r1_state.metadata: ', u1_r1_state.metadata)
    # print('  u1_r1_state.values: ', u1_r1_state.values)
    print('\nu1_r1_state.values -> messages:')
    for index, message in enumerate(u1_r1_state.values['messages'], start=1):
        print(f"[{index}] {message.content}")

    print("\n-------- user-1 chat-round-2 ------------")
    msg_2 = [{'role': 'user', 'content': '我们刚才聊了什么？'}]
    u1_r2 = compile_graph.invoke(input={"messages": msg_2}, config=u1_config)
    # print(u1_r2)
    for msg in u1_r2['messages']:
        # print(msg.content)
        msg.pretty_print()
    u1_r2_state = compile_graph.get_state(config=u1_config)
    # print("\n---> u1_r2_state show:")
    # print('  u1_r2_state.config: ', u1_r2_state.config)
    # print('  u1_r2_state.metadata: ', u1_r2_state.metadata)
    # print('  u1_r2_state.values: ', u1_r2_state.values)
    print('\nu1_r1_state.values -> messages:')
    for index, message in enumerate(u1_r2_state.values['messages'], start=1):
        print(f"[{index}] {message.content}")


# %% ======================= Tool调用 =======================
def chatbot_tool_usage_manual():
    """
    展示tool调用，这里先手动实现 tool 调用.
    """
    # 定义 tool
    @tool(description="使用龙球(DragonBall)算法计算两个数字的结果")
    def dragon_ball_algorithm_tool(x: Annotated[int, "第一个数字"], y: Annotated[int, "第二个数字"]) -> int:
        return x + y + 1

    # 初始化 ChatLLM，并绑定 tool
    client_chat = get_client_chat()
    client_chat_tool = client_chat.bind_tools(tools=[dragon_ball_algorithm_tool])

    # 定义 StateGraph
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    #  定义 chatbot 节点
    def chatbot_node(state: State):
        print(f"--> chatbot_node start...")
        print(f"  state: {state}")
        response = client_chat_tool.invoke(input=state["messages"])
        print(f"<-- chatbot_node end.")
        return {"messages": [response]}

    # 定义一个 Tools 调用节点
    class CustomToolNode:
        """
        A node that runs the tools requested in the last AIMessage.
        此示例来自官方文档 [Create a function to run the tools](https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/#5-create-a-function-to-run-the-tools)
        """
        def __init__(self, tools: List[BaseTool]) -> None:
            # 使用dict，存储多个工具，将工具名称映射到工具对象
            self.tools_by_name = {t.name: t for t in tools}

        def __call__(self, inputs: State):
            # 尝试从 state 中获取最后一个元素
            if messages := inputs.get("messages", []):
                message = messages[-1]
            else:
                raise ValueError("No message found in input")
            # 触发 tool 调用时，最后一个 message 应该是 AIMessage，并且有 tool_calls 属性 —— 不过这个判断不放在这里，而是放在了 conditional_edge 中
            assert isinstance(message, AIMessage), "Last message is not an AIMessage"
            outputs = []
            for tool_call in message.tool_calls:
                tool_func = self.tools_by_name[tool_call["name"]]
                tool_result = tool_func.invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}

    # 定义判断是否调用 tool 节点的 条件边
    def route_tools(state: State):
        """
        Use in the conditional_edge to route to the ToolNode if the last message has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        assert isinstance(ai_message, AIMessage), "Last message is not an AIMessage"
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    # 构建图
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", CustomToolNode(tools=[dragon_ball_algorithm_tool]))
    graph_builder.add_conditional_edges("chatbot", route_tools)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile(name="ChatbotWithToolGraph")

    # 调用
    input_msgs = [
        SystemMessage(content='你是一个算术专家'),
        HumanMessage(content='请使用龙球(DragonBall)算法计算一下 2019 和 2022 的结果'),
    ]
    res = graph.invoke(input={"messages": input_msgs})
    for msg in res['messages']:
        msg.pretty_print()

# %%
def chatbot_tool_usage_prebuilt():
    """
    还是上面的例子，不过这次使用 LangGraph 提供预构建的 ToolNode 和 tools_condition
    """
    # 定义 tool
    @tool(description="使用龙球(DragonBall)算法计算两个数字的结果")
    def dragon_ball_algorithm_tool(x: Annotated[int, "第一个数字"], y: Annotated[int, "第二个数字"]) -> int:
        return x + y + 1

    # ------ 工具上下文使用 ------
    # 使用 dataclass 自定义一个上下文，里面包含需要使用的字段信息
    @dataclass
    class Context:
        user_id: str

    @tool(description="获取工具上下文")
    def get_tool_context(runtime: ToolRuntime[Context]) -> str:
        # 从工具运行时获取工具上下文里的字段
        user_id = runtime.context.user_id
        preferences: str = "The user prefers you to write a brief and polite email."
        if runtime.store:
            if memory := runtime.store.get(("users",), user_id):
                preferences = memory.value["preferences"]
        return preferences
    # ------ 工具上下文使用 ------

    # 初始化 ChatLLM，并绑定 tool
    client_chat = get_client_chat()
    client_chat_tool = client_chat.bind_tools(tools=[dragon_ball_algorithm_tool])

    # 定义 StateGraph
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    #  定义 chatbot 节点
    def chatbot_node(state: State):
        print(f"--> chatbot_node start...")
        print(f"  state: {state}")
        response = client_chat_tool.invoke(input=state["messages"])
        print(f"<-- chatbot_node end.")
        return {"messages": [response]}

    # 定义 tools 调用节点，直接使用 LangGraph 提供的 ToolNode 类 —— 这个类内部的实现就类似于上面的 CustomToolNode
    tool_node = ToolNode(tools=[dragon_ball_algorithm_tool])

    # 构建图
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", tool_node)
    # 判断是否调用 tool 节点的 条件边 直接使用 tools_condition # TODO 这个条件判断逻辑有待仔细研究
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile(name="ChatbotWithToolGraph")

    # 调用
    input_msgs = [
        SystemMessage(content='你是一个算术专家'),
        HumanMessage(content='请使用龙球(DragonBall)算法计算一下 2019 和 2022 的结果'),
    ]
    res = graph.invoke(input={"messages": input_msgs})
    for msg in res['messages']:
        msg.pretty_print()


# %% ======================= ReAct Agent 生成 =======================
def react_agent_usage():
    """
    API文档: [create_react_agent](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.chat_agent_executor.create_react_agent)
    注意：create_react_agent() 这个API 在 LangGraph v1.0 版本被标记为了废弃，后续推荐直接使用 langchain.agents 里提供的 create_agent()
    """
    @tool(description="使用龙球(DragonBall)算法计算两个数字的结果")
    def dragon_ball_algorithm(x: Annotated[int, "第一个数字"], y: Annotated[int, "第二个数字"]) -> int:
        return x + y + 1

    @tool(description="检查龙球(DragonBall)算法的结果是否正确")
    def dragon_ball_check(x: Annotated[int, "第一个数字"], y: Annotated[int, "第二个数字"], z: Annotated[int, "结果数字"]) -> int:
        return x + y + 1 == z

    tools = [dragon_ball_algorithm, dragon_ball_check]
    # 可以试下不传入检查结果的工具，模型会给出不一样的回答
    # tools = [dragon_ball_algorithm]

    tool_node = ToolNode(tools=tools)
    client_chat = get_client_chat()
    client_chat_tool = client_chat.bind_tools(tools=tools)
    memory = MemorySaver()

    # 创建 Agent 的图，这里使用默认的 AgentState，当然也可以自定义
    # tools 参数可以使用 ToolNode，也可以直接使用 List[Tool]
    agent = create_react_agent(name='ReAct-Agent', model=client_chat_tool, tools=tool_node, checkpointer=memory)
    print(type(agent))
    print(agent.name)

    # 输入
    input_msgs = [
        SystemMessage(content='你是一个算术专家'),
        HumanMessage(content='请使用龙球(DragonBall)算法计算一下 2019 和 2022 的结果，并检查算法的结果是否正确'),
    ]
    config = {"configurable": {"thread_id": "1"}}

    # 调用
    res = agent.invoke(input={"messages": input_msgs}, config=config)
    print(f"type(res): {type(res)}")    # <class 'langgraph.pregel.io.AddableValuesDict'>
    print(res.keys())    # dict_keys(['messages'])
    print(res)
    print("-------------------------------")
    for msg in res['messages']:
        msg.pretty_print()


# %% ======================= Graph Stream =======================
def graph_stream_usage():
    """
    展示 Graph 的 stream 接口使用。
    """
    @tool(description="使用龙球(DragonBall)算法计算两个数字的结果")
    def dragon_ball_algorithm(x: Annotated[int, "第一个数字"], y: Annotated[int, "第二个数字"]) -> int:
        return x + y + 1

    @tool(description="检查龙球(DragonBall)算法的结果是否正确")
    def dragon_ball_check(x: Annotated[int, "第一个数字"], y: Annotated[int, "第二个数字"], z: Annotated[int, "结果数字"]) -> int:
        return x + y + 1 == z

    tools = [dragon_ball_algorithm, dragon_ball_check]
    tool_node = ToolNode(tools=tools)
    client_chat = get_client_chat()
    client_chat_tool = client_chat.bind_tools(tools=tools)
    memory = MemorySaver()
    agent = create_react_agent(name='ReAct-Agent', model=client_chat_tool, tools=tool_node, checkpointer=memory)

    # 输入
    input_msgs = [
        SystemMessage(content='你是一个算术专家'),
        HumanMessage(content='请使用龙球(DragonBall)算法计算一下 2019 和 2022 的结果，并检查算法的结果是否正确'),
    ]
    config = {"configurable": {"thread_id": "1"}}

    # Stream调用
    # 可接受的模式有：["values", "updates", "debug", "messages", "custom"]，参见 langgraph.types.StreamMode

    # None：也是默认模式，此时每次输出一个 Message
    print("------- Stream with None -------")
    for chunk in agent.stream(input={"messages": input_msgs}, config=config):
        # print(f"type(chunk): {type(chunk)}")   # <class 'langgraph.pregel.io.AddableUpdatesDict'>
        print(chunk.keys())
        print(chunk)

    # values 模式：在图中的每个步骤之后流式传输状态的完整值
    print("\n------- Stream with values -------")
    for chunk in agent.stream(input={"messages": input_msgs}, config=config, stream_mode="values"):
        print(chunk.keys())
        print(chunk)

    # updates 模式：在图中的每个步骤之后将更新流式传输到状态。
    print("\n------- Stream with update -------")
    for chunk in agent.stream(input={"messages": input_msgs}, config=config, stream_mode="updates"):
        print(chunk.keys())
        print(chunk)

    # messages 模式：记录每个 messages 中的增量 token。
    print("\n------- Stream with messages -------")
    for token, metadata in agent.stream(input={"messages": input_msgs}, config=config, stream_mode="messages"):
        print("Token: ", token)
        print("Metadata: ", metadata)

    # debug 模式：在整个图的执⾏过程中流式传输尽可能多的信息，主要⽤于调试程序。
    # print("------- Stream with debug -------")
    # for chunk in agent.stream(input={"messages": input_msgs}, config=config, stream_mode="debug"):
    #     print(chunk)

    # custom 模式：⾃定义流，通过 LangGraph 的 StreamWriter ⽅法
    # print("------- Stream with custom -------")
    # for chunk in agent.stream(input={"messages": input_msgs}, config=config, stream_mode="custom"):
    #     print(chunk)


# %% ======================= Graph Node 自定义 =======================
def graph_custom_node_with_class_usage():
    """
    展示如何自定义 Graph 里的node。
    参考官方文档[Graph API concepts](https://langchain-ai.github.io/langgraph/concepts/low_level/).
    LangGraph 里的 Node，一般是一个 Python function，入参是自定义的 State，返回的是更新后的 State 或者 State 里的某个key的值。
    但是有时候为了执行一些复杂逻辑，Node 也可以定义成一个 class，此时有两种做法：
    1. 类似于 chatbot_tool_usage_manual 示例中那样，定义一个 class，需要实现其中的 __call__ 方法，使其称为一个 Callable 对象，这种方式比较简单
    2. 类似于 chatbot_tool_usage_prebuilt 示例中那样，参考其中 ToolNode 类的实现，继承 langgraph 提供的 RunnableCallable 抽象类，实现
    """
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    class CustomNode:
        def __init__(self, name: str) -> None:
            self.name = f"<CustomNode:{name}>"

        def __call__(self, inputs: State):
            print(f"{self.name} called with inputs: {inputs}")
            return {"messages": [f"{self.name} says: Hello!"]}

    class RunnableNode(RunnableCallable):
        def __init__(self, name: str) -> None:
            name = f"<RunnableNode:{name}>"
            # RunnableCallable 的 __init__ 方法里，只有 func 参数是必须的，其他都可选，这里是仿照的 ToolsNode 的实现
            super().__init__(func=self._func, name=name)
            self.name = name

        def _func(self, inputs: State):
            """同步调用函数"""
            print(f"{self.name} called with inputs: {inputs}")
            return {"messages": [f"{self.name} says: Welcome!"]}

    graph = StateGraph(state_schema=State)
    graph.add_node(node="custom_node", action=CustomNode(name="SomeNode"))
    graph.add_node(node="runnable_node", action=RunnableNode(name="SomeRunNode"))
    graph.set_entry_point("custom_node")
    graph.add_edge("custom_node", "runnable_node")
    graph.set_finish_point("runnable_node")
    compile_graph: CompiledStateGraph = graph.compile(name='GraphWithCustomNode')

    res = compile_graph.invoke(input={"messages": [HumanMessage(content="Hi")]})
    print(res)


def show_graph(graph: CompiledStateGraph):
    """
    绘制LangGraph 的 Graph 图结构.
    参考官方文档[Graph API -> Use the graph API -> Visualize your graph](https://docs.langchain.com/oss/python/langgraph/use-graph-api#visualize-your-graph)
    """
    try:
        from IPython.display import Image, display
        # display(Image(graph.get_graph().draw_mermaid_png()))
        img = Image(graph.get_graph().draw_mermaid_png())
        with open(f"{graph.get_name()}.png", 'wb') as f:
            f.write(img.data)
    except Exception as e:
        # This may require some extra dependencies and is optional
        print(e)


# %% ======================= Main =======================
def main():
    stateful_graph_usage()
    # message_graph_usage()
    # graph_conditional_usage()
    # graph_checkpoint_usage()
    # graph_store_usage()
    # graph_dynamic_interrupt_usage()
    # graph_fixed_interrupt_usage()
    # chatbot_example()
    # chatbot_tool_usage_manual()
    # chatbot_tool_usage_prebuilt()
    # react_agent_usage()
    # graph_stream_usage()
    # graph_custom_node_with_class_usage()


if __name__ == '__main__':
    main()
