"""
简单研究 LangGraph 的使用.
主要参考了如下官方文档：
- [LangGraph Glossary](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
首先需要明确的是，LangGraph 不依赖 Langchain-Core 或者 Langchain，因此下面的研究都使用一个简单的Python Callable 对象
来代替实际中的 Langchain-Core/Langchain 里的 Runnable/LLM/ChatModel/Chain 对象。
"""
from typing import Annotated, List, TypedDict, Dict, Union
# from typing_extensions import TypedDict
from langgraph.graph import Graph, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.constants import START, END
from langgraph.graph.message import MessageGraph, MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage
from langchain_openai.chat_models import ChatOpenAI

API_KEY = 'Random'
LLM_URL = 'http://172.16.0.32:10086/v1'

# ======================= 无状态图 构建 =======================
def stateless_graph_usage():
    # 创建一个 Graph
    graph = Graph()
    # 定义节点
    graph.add_node(node="hello", action=lambda _: "Hello, world !")
    graph.add_node(node="welcome", action=lambda _: "Welcome LangGraph !")
    graph.add_node(node="bye", action=lambda _: "Goodbye !")

    # 定义边
    graph.add_edge("hello", "welcome")
    graph.add_edge("welcome", "bye")

    # 定义起始点和结束点
    graph.set_entry_point('hello')
    graph.set_finish_point('bye')

    # 编译并执行
    compile_graph = graph.compile(name='StatelessGraph')
    print(compile_graph.config_specs)
    print(compile_graph.config_type)
    result = compile_graph.invoke(input={'key1': 'value1', 'key2': 'value2'})
    print(result)


# ======================= 简单有状态图 构建 =======================
def stateful_graph_usage():
    # 1. 首先定义整个 Graph 的状态表示，可以直接用dict，也可以用 TypedDict，或者是 Pydantic的 BaseModel —— 状态表示完全由用户自定义
    class SimpleState(TypedDict):
        messages: List[str]
        count: int

    # 2. 定义Node里要运行的Python函数
    def greet_node(state: SimpleState) -> SimpleState:
        state["messages"].append("Hello")
        print(f"--> greet_node running...")
        return state

    def increment_node(state):
        state["count"] += 1
        print(f"--> increment_node running...")
        return state

    def something_node(state):
        state["messages"].append("Something")
        state["count"] += 2
        print(f"--> something_node running...")
        return state

    # 3. 使用StateGraph 构建 Graph，初始化参数必须传入自定义的 State 类
    graph = StateGraph(state_schema=SimpleState)

    # 4. 使用 add_node 方法添加节点，add_node 方法有多个重载，注意选择
    graph.add_node(node="greet_node", action=greet_node)
    graph.add_node(node="increment_node", action=increment_node)
    graph.add_node(node="something_node", action=something_node)

    # 5. 定义边
    graph.set_entry_point("greet_node")
    graph.add_edge("greet_node", "increment_node")
    graph.add_edge("increment_node", "something_node")
    graph.set_finish_point("something_node")

    # 6. 编译构建 Graph，返回的是 CompiledStateGraph 对象
    compile_graph: CompiledStateGraph = graph.compile(name='SimpleStateGraph')
    # 可以查看具体的图结构
    # graph_picture = compile_graph.get_graph()
    print(compile_graph.config_specs)
    print(compile_graph.name)
    print(compile_graph.get_name())

    # 7. 使用 Graph
    # 运行Graph: 调用 invoke/ainvoke; stream/astream 方法
    res = compile_graph.invoke(input={"messages": [], "count": 0})
    print(res)


# ======================= 基于条件动态执行有状态图 =======================
def graph_conditional_usage():
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
            print(f"==> conditional_edge_check: -> reset_count_node")
            return "reset_count_node"
        else:
            print(f"==> conditional_edge_check: -> END")
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


# ======================= Checkpoint Graph 构建 =======================
def graph_checkpoint_usage():
    # 为了展示 checkpoint 的效果，定义的 state 对象里，需要有一个 reducer 函数 —— 这里使用了现成的 add_messages
    def num_reducer(prev_num: List[int], curr_num: List[int]) -> List[int]:
        return prev_num + curr_num

    class ReduceState(TypedDict):
        # add_messages 函数是专门为 BasedMessage 设置的 reducer，它会自动将字符串用 HumanMessage 包装起来
        messages: Annotated[List[str], add_messages]
        num: Annotated[List[int], num_reducer]

    def greet_node(state: ReduceState) -> Dict[str, List[str]]:
        state_msg = 'Hello'
        state_num = 0
        if len(state['messages']) > 0:
            # state_msg = state['messages'][-1]
            state_msg = state['messages'][-1].content
        if len(state["num"]) > 0:
            state_num = state['num'][-1] + 1
        new_message = f"{state_msg}[{state_num}]"
        print(f"--> greet_node running...")
        return {'messages': [new_message]}

    def increment_node(state: ReduceState) -> Dict[str, List[int]]:
        state_num = 0
        if len(state["num"]) > 0:
            state_num = state['num'][-1] + 1
        print(f"--> increment_node running...")
        return {'num': [state_num]}

    graph = StateGraph(state_schema=ReduceState)
    graph.add_node(node="greet_node", action=greet_node)
    graph.add_node(node="increment_node", action=increment_node)
    graph.set_entry_point("greet_node")
    graph.add_edge("greet_node", "increment_node")
    graph.set_finish_point("increment_node")

    # 这个就是 checkpoint 对象，在compile的时候传入
    memory = MemorySaver()
    compile_graph: CompiledStateGraph = graph.compile(name='StateGraphWithCheckpoint', checkpointer=memory)

    # 调用的时候传入一个config参数，代表当前用户身份
    config_u1 = {"configurable": {"thread_id": "user-1"}}
    print("-------- user-1 call-1 ------------")
    u1_r1 = compile_graph.invoke(input={"messages": [], "num": []}, config=config_u1)
    # print(type(u1_r1))  # <class 'langgraph.pregel.io.AddableValuesDict'>
    # print(u1_r1)
    for msg, num in zip(u1_r1["messages"], u1_r1["num"]):
        print(f"num: {num}; msg: {msg}")
    # 获取当前的状态，必须要使用 invoke 时同样的 config
    u1_r1_state = compile_graph.get_state(config=config_u1)
    # print(type(u1_r1_state))  # <class 'langgraph.types.StateSnapshot'>
    # print(u1_r1_state)
    print("---> u1_r1_state show:")
    # print('  u1_r1_state.config: ', u1_r1_state.config)
    # print('  u1_r1_state.metadata: ', u1_r1_state.metadata)
    print('  u1_r1_state.values: ', u1_r1_state.values)  # 这个就是当前 state 对象的值，应该和 u1_r1 的内容是一样的

    print("-------- user-1 call-2 ------------")
    u1_r2 = compile_graph.invoke(input={"messages": ["Call-2"], "num": [10]}, config=config_u1)
    # print(u1_r2)
    for msg, num in zip(u1_r2["messages"], u1_r2["num"]):
        print(f"num: {num}; msg: {msg}")
    u1_r2_state = compile_graph.get_state(config=config_u1)
    print("---> u1_r2_state show:")
    # print('  u1_r2_state.config: ', u1_r2_state.config)
    # print('  u1_r2_state.metadata: ', u1_r2_state.metadata)
    print('  u1_r2_state.values: ', u1_r2_state.values)

    # ---- 获取历史状态，这也是 TimeTravel 的原理，获取历史状态，然后Replay ----
    history_states = compile_graph.get_state_history(config_u1)
    for state in history_states:
        # print(type(state))  # <class 'langgraph.types.StateSnapshot'>
        # print(state)
        print("---> state show:")
        print('  state.metadata: ', state.metadata)  # 这个有用
        print('  state.values: ', state.values)  # 这个有用
        # print('  state.config: ', state.config)
        # print('  state.tasks: ', state.tasks)
        # print('  state.next: ', state.next)
        # print('  state.parent_config: ', state.parent_config)

    # ---- 获取所有 checkpoint 列表 ----
    checkpoint_iter = memory.list(config=config_u1)
    # 下面展示的checkpoint顺序是倒序的
    for checkpoint in checkpoint_iter:
        # print(type(checkpoint))   # <class 'langgraph.checkpoint.base.CheckpointTuple'>
        # print(checkpoint)
        print("---> checkpoint show:")
        # print('  checkpoint.config: ', checkpoint.config)
        print('  checkpoint.metadata: ', checkpoint.metadata)  # 这个信息最有用
        # print('  checkpoint.pending_writes: ', checkpoint.pending_writes)
        # print('  checkpoint.checkpoint: ', checkpoint.checkpoint)
        # print('  checkpoint.parent_config: ', checkpoint.parent_config)


# ======================= Interrupt/Command机制 =======================
def graph_interrupt_usage():
    pass


# ======================= 结合 LangChain 的 ChatBot 案例 =======================
def chatbot_example():
    client_chat = ChatOpenAI(
        openai_api_key=API_KEY,
        openai_api_base=LLM_URL,
        model_name='Qwen2.5-32B-Instruct'
    )
    # res = client_chat.invoke(input=[{'role': 'user', 'content': '你好，可以和我聊聊历史吗？'}])
    # print(res.content)

    class MsgState(TypedDict):
        messages: Annotated[list[Union[str, BaseMessage]], add_messages]

    def chatbot(state: MsgState):
        return {"messages": [client_chat.invoke(input=state["messages"])]}

    graph = StateGraph(MsgState)
    graph.add_node(node='chatbot', action=chatbot)
    graph.set_entry_point('chatbot')
    graph.set_finish_point('chatbot')

    memory = MemorySaver()
    compile_graph = graph.compile(name='ChatBot', checkpointer=memory)

    config_u1 = {"configurable": {"thread_id": "user-1"}}
    print("-------- user-1 chat-round-1 ------------")
    messages_r1 = [{'role': 'user', 'content': '你好，可以和我聊聊历史吗？'}]
    u1_r1 = compile_graph.invoke(input={"messages": messages_r1}, config=config_u1)
    # print(type(u1_r1))  # <class 'langgraph.pregel.io.AddableValuesDict'>
    # print(u1_r1['messages'])
    for message in u1_r1['messages']:
        print(message.content)
    u1_r1_state = compile_graph.get_state(config=config_u1)
    print("---> u1_r1_state show:")
    # print('  u1_r1_state.config: ', u1_r1_state.config)
    # print('  u1_r1_state.metadata: ', u1_r1_state.metadata)
    # print('  u1_r1_state.values: ', u1_r1_state.values)
    print('u1_r1_state.values -> messages:')
    for message in u1_r1_state.values['messages']:
        print(message.content)

    print("-------- user-1 chat-round-2 ------------")
    messages_r2 = [{'role': 'user', 'content': '我们刚才聊了什么？'}]
    u1_r2 = compile_graph.invoke(input={"messages": messages_r2}, config=config_u1)
    # print(u1_r2)
    for message in u1_r2['messages']:
        print(message.content)
    u1_r2_state = compile_graph.get_state(config=config_u1)
    print("---> u1_r2_state show:")
    # print('  u1_r2_state.config: ', u1_r2_state.config)
    # print('  u1_r2_state.metadata: ', u1_r2_state.metadata)
    # print('  u1_r2_state.values: ', u1_r2_state.values)
    print('u1_r1_state.values -> messages:')
    for message in u1_r2_state.values['messages']:
        print(message.content)


def main():
    # stateful_graph_usage()
    # graph_conditional_usage()
    graph_checkpoint_usage()
    # chatbot_example()


if __name__ == '__main__':
    main()
