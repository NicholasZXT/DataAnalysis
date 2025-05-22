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
from langgraph.constants import START, END
from langgraph.graph import Graph, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import MessageGraph, MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent, InjectedState, InjectedStore
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
# --- langchain 依赖 ---
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import BaseTool, Tool, StructuredTool, tool
# --- 其他依赖 ---
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

# ======================= 无状态图 构建 =======================
def stateless_graph_usage():
    # 创建一个 Graph，这个Graph类不接受任何初始化参数，所以说它是无状态的。
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
        print(f"--> greet_node start...")
        print(f"  state: {state}")
        state["messages"].append("Hello")
        print(f"<-- greet_node end.")
        return state

    def increment_node(state):
        print(f"--> increment_node start...")
        print(f"  state: {state}")
        state["count"] += 1
        print(f"<-- increment_node end.")
        return state

    def something_node(state):
        print(f"--> something_node start...")
        print(f"  state: {state}")
        state["messages"].append("Something")
        state["count"] += 2
        print(f"<-- something_node end.")
        return state

    def partial_node(state):
        print(f"--> partial_node start...")
        print(f"  state: {state}")
        print(f"<-- partial_node end.")
        # 只返回状态部分的key也可以，不过此时如果对应的 key 没有设置 reducer 函数的话，默认会覆盖该 key 的内容
        return {'count': 10}

    # 3. 使用StateGraph 构建 Graph，初始化参数必须传入自定义的 State 类
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
    input = {"messages": [], "count": 0}
    print(f"Graph input: {input}")
    res = compile_graph.invoke(input=input)
    print(f"final state: {res}")


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


# ======================= Graph Checkpoint 使用 =======================
def graph_checkpoint_usage():
    # 为了展示 checkpoint 的效果，定义的 state 对象里，每个属性需要有一个 reducer 函数，这里使用了两种：
    # 1. 自定义 reducer
    # 2. LangGraph提供的 add_messages，该reducer函数是专门为 BasedMessage(langchain_core提供) 设置的 reducer，
    #    它会自动将字符串用 HumanMessage 包装起来，并追加到末尾，因此它要求对应的属性是一个 List[BasedMessage] 类型
    # Reducer 函数最大的作用是用于保存历史对话记录
    def num_reducer(prev_num: List[int], curr_num: List[int]) -> List[int]:
        return prev_num + curr_num

    class ReduceState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]
        num: Annotated[List[int], num_reducer]

    def greet_node(state: ReduceState) -> Dict[str, List[str]]:
        print(f"--> greet_node start...")
        print(f"  state: {state}")
        # greet_node 对应初始空状态的默认信息
        state_msg, state_num = 'Hello', 0
        if len(state['messages']) > 0:
            # state_msg = state['messages'][-1]
            state_msg = state['messages'][-1].content
        if len(state["num"]) > 0:
            state_num = state['num'][-1] + 1
        new_message = f"{state_msg}[{state_num}]"
        print(f"<-- greet_node end.")
        return {'messages': [new_message]}

    def increment_node(state: ReduceState) -> Dict[str, List[int]]:
        print(f"--> increment_node start...")
        print(f"  state: {state}")
        # increment_node 对应初始空状态的默认信息
        state_num = 0
        if len(state["num"]) > 0:
            state_num = state['num'][-1] + 1
        print(f"<-- increment_node end.")
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

    print("-------- user-1 call-1 ------------")
    u1_input_1 = {"messages": [], "num": []}
    # 调用的时候传入一个config字段，key 必须是 configurable，里面设置一个 thread_id，用于表示当前用户身份
    config_u1 = {"configurable": {"thread_id": "user-1"}}
    print(f"u1_input_1: {u1_input_1}")
    u1_r1 = compile_graph.invoke(input=u1_input_1, config=config_u1)
    # print(type(u1_r1))  # <class 'langgraph.pregel.io.AddableValuesDict'>
    # print(f"u1_r1 state: {u1_r1}")
    print(f"==> u1_r1 state:")
    for msg, num in zip(u1_r1["messages"], u1_r1["num"]):
        msg.pretty_print()
        print(f"num: {num}")
    # 获取当前的状态，必须要使用 invoke 时同样的 config
    u1_r1_state = compile_graph.get_state(config=config_u1)
    # print(type(u1_r1_state))  # <class 'langgraph.types.StateSnapshot'>
    # print(u1_r1_state)
    print("\n==> u1_r1_state show:")
    # print('  u1_r1_state.config: ', u1_r1_state.config)
    # print('  u1_r1_state.metadata: ', u1_r1_state.metadata)
    print('  u1_r1_state.values: ', u1_r1_state.values)  # 这个就是当前 state 对象的值，应该和 u1_r1 的内容是一样的

    print("\n-------- user-1 call-2 ------------")
    u1_input_2 = {"messages": ["Call-2"], "num": [10]}
    print(f"u1_input_2: {u1_input_2}")
    u1_r2 = compile_graph.invoke(input=u1_input_2, config=config_u1)
    # print(f"u1_r2 state: {u1_r2}")
    print("==> u1_r2 state:")
    for msg, num in zip(u1_r2["messages"], u1_r2["num"]):
        msg.pretty_print()
        print(f"num: {num}")
    u1_r2_state = compile_graph.get_state(config=config_u1)
    print("\n==> u1_r2_state show:")
    # print('  u1_r2_state.config: ', u1_r2_state.config)
    # print('  u1_r2_state.metadata: ', u1_r2_state.metadata)
    print('  u1_r2_state.values: ', u1_r2_state.values)

    # ---- 获取历史状态，这也是 TimeTravel 的原理，获取历史状态，然后Replay ----
    print("\n-------- History State ------------")
    history_states = compile_graph.get_state_history(config_u1)
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

    # ---- 获取所有 checkpoint 列表 ----
    print("\n-------- Checkpoints ------------")
    checkpoint_iter = memory.list(config=config_u1)
    # 下面展示的checkpoint顺序是倒序的 ---------- KEY
    for checkpoint in checkpoint_iter:
        # print(type(checkpoint))   # <class 'langgraph.checkpoint.base.CheckpointTuple'>
        # print(checkpoint)
        print("==> checkpoint show:")
        # print('  checkpoint.config: ', checkpoint.config)
        print('  checkpoint.metadata: ', checkpoint.metadata)  # 这个信息最有用
        # print('  checkpoint.pending_writes: ', checkpoint.pending_writes)
        # print('  checkpoint.checkpoint: ', checkpoint.checkpoint)
        # print('  checkpoint.parent_config: ', checkpoint.parent_config)


# ======================= Interrupt/Command机制 =======================
def graph_interrupt_usage():
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
            value = {"state.num": ""}
        print(f"  ==> greet_node is waiting for human response with value: {value}")
        # 使用 interrupt 函数打断图的执行，等待人工输入
        human_response = interrupt(value=value)
        print(f"  <== greet_node received human response: {human_response}")
        print(f"<-- greet_node end.")
        return {'human_msg': human_response["human_msg"]}

    def increment_node(state: HumanInterruptState) -> Dict[str, List[int]]:
        print(f"--> increment_node start...")
        state_num = 0
        if len(state["num"]) > 0:
            state_num = state['num'][-1] + 1
        print(f"<-- increment_node end.")
        return {'num': [state_num]}

    graph = StateGraph(state_schema=HumanInterruptState)
    graph.add_node(node="greet_node", action=greet_node)
    graph.add_node(node="increment_node", action=increment_node)
    graph.set_entry_point("greet_node")
    graph.add_edge("greet_node", "increment_node")
    graph.set_finish_point("increment_node")

    memory = MemorySaver()
    compile_graph: CompiledStateGraph = graph.compile(name='StateGraphWithHumanInterrupt', checkpointer=memory)

    config_u1 = {"configurable": {"thread_id": "user-1"}}
    u1_input = {"messages": '', "num": []}
    print(f"u1_input = {u1_input}")
    u1_r1 = compile_graph.invoke(input=u1_input, config=config_u1)
    print(f"u1_r1 state: {u1_r1}")
    print("--------------------")
    resume = {'human_msg': "hello world"}
    print(f"resume: {resume}")
    u1_r1_command = Command(resume=resume)
    u1_r1_continue = compile_graph.invoke(input=u1_r1_command, config=config_u1)
    print(f"u1_r1_continue: {u1_r1_continue}")


# ======================= 结合 LangChain 的 ChatBot 案例 =======================
def chatbot_example():
    client_chat = get_client_chat()
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
    compile_graph = graph.compile(name='ChatBotGraph', checkpointer=memory)

    config_u1 = {"configurable": {"thread_id": "user-1"}}
    print("-------- user-1 chat-round-1 ------------")
    msg_1 = [{'role': 'user', 'content': '你好，可以和我聊聊历史吗？'}]
    u1_r1 = compile_graph.invoke(input={"messages": msg_1}, config=config_u1)
    # print(type(u1_r1))  # <class 'langgraph.pregel.io.AddableValuesDict'>
    # print(u1_r1['messages'])
    for msg in u1_r1['messages']:
        # print(msg.content)
        msg.pretty_print()
    u1_r1_state = compile_graph.get_state(config=config_u1)
    # print("\n---> u1_r1_state show:")
    # print('  u1_r1_state.config: ', u1_r1_state.config)
    # print('  u1_r1_state.metadata: ', u1_r1_state.metadata)
    # print('  u1_r1_state.values: ', u1_r1_state.values)
    print('\nu1_r1_state.values -> messages:')
    for index, message in enumerate(u1_r1_state.values['messages'], start=1):
        print(f"[{index}] {message.content}")

    print("\n-------- user-1 chat-round-2 ------------")
    msg_2 = [{'role': 'user', 'content': '我们刚才聊了什么？'}]
    u1_r2 = compile_graph.invoke(input={"messages": msg_2}, config=config_u1)
    # print(u1_r2)
    for msg in u1_r2['messages']:
        # print(msg.content)
        msg.pretty_print()
    u1_r2_state = compile_graph.get_state(config=config_u1)
    # print("\n---> u1_r2_state show:")
    # print('  u1_r2_state.config: ', u1_r2_state.config)
    # print('  u1_r2_state.metadata: ', u1_r2_state.metadata)
    # print('  u1_r2_state.values: ', u1_r2_state.values)
    print('\nu1_r1_state.values -> messages:')
    for index, message in enumerate(u1_r2_state.values['messages'], start=1):
        print(f"[{index}] {message.content}")


# ======================= Tool调用 =======================
def chatbot_tool_usage_manual():
    """
    展示tool调用，这里先手动实现 tool 调用
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
    def chatbot(state: State):
        return {"messages": [client_chat_tool.invoke(state["messages"])]}

    # 定义一个 Tool 调用节点
    class CustomToolNode:
        """
        A node that runs the tools requested in the last AIMessage.
        此示例来自官方文档 [Create a function to run the tools](https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/#5-create-a-function-to-run-the-tools)
        """
        def __init__(self, tools: List[BaseTool]) -> None:
            self.tools_by_name = {t.name: t for t in tools}

        def __call__(self, inputs: State):
            # 尝试从 state 中获取最后一个元素
            if messages := inputs.get("messages", []):
                message = messages[-1]
            else:
                raise ValueError("No message found in input")
            # 触发 tool 调用时，最后一个 message 应该是 AIMessage，并且有  tool_calls 属性 —— 这个判断不放在这里，而是放在了 conditional_edge 中
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
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    # 构建图
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
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

def chatbot_tool_usage_prebuilt():
    """
    还是上面的例子，不过这次使用 LangGraph 提供预构建的 ToolNode 和 tools_condition
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
    def chatbot(state: State):
        return {"messages": [client_chat_tool.invoke(state["messages"])]}

    # 定义 tool 调用节点，直接使用 LangGraph 提供的 ToolNode 类
    tool_node = ToolNode(tools=[dragon_ball_algorithm_tool])

    # 判断是否调用 tool 节点的 条件边 直接使用 tools_condition

    # 构建图
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
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


def main():
    # stateful_graph_usage()
    # graph_conditional_usage()
    # graph_checkpoint_usage()
    # graph_interrupt_usage()
    # chatbot_example()
    # chatbot_tool_usage_manual()
    chatbot_tool_usage_prebuilt()


if __name__ == '__main__':
    main()
