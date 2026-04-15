[TOC]

# V1.0升级说明

LangChain & LangGraph 框架于 2025-10-18 **同时正式发布了 v1.0 版本**，相比与之前的 v0.3/v0.6 版本，有了不少重大升级。

LangChain & LangGraph 框架 v1.0 的官方（Python）文档地址也变更为 [LangChain Docs](https://docs.langchain.com/).

| 版本         | LangChain-Core | LangChain | LangGraph |
|------------|----------------|-----------|-----------|
| v0.x最后一个版本 | 0.3.80         | 0.3.27    | 0.6.11    |

`langchain-community`包目前还没有升级到v1.x版本，好像变化不大。

------
## LangChain

官方文档：

- [Release -> What's new in LangChain v1](https://docs.langchain.com/oss/python/releases/langchain-v1)，LangChain v1.0 的发布说明
- [LangChain v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1)，从 v0.3 版本的迁移说明

LangChain v0.3版本，官方文档介绍时的第一句话是：

> **LangChain** is a framework for developing applications powered by large language models (LLMs).

而到了LangChain v1.0版本，官方文档介绍的第一句话是：

> LangChain is the easiest way to start building agents and applications powered by LLMs.

可以看出，**LangChain v1.0 版本的重心应该是向Agent开发框架倾斜了**。

根据官方文档 [Release -> What's new in LangChain v1](https://docs.langchain.com/oss/python/releases/langchain-v1) 的说明，LangChain v1.0 的重大改变如下：

> We’ve streamlined the framework around three core improvements：

**一、核心架构变化**

**（1）LangGraph 成为基石**

- LangChain-v0.3.x和 LangGraph-v0.6.x版本是相对独立的两个项目，两者没有太多交集和耦合，这也导致LangChain/LangGraph里分别有各自构建Agent的API，过于复杂；
- 但是从v1.0版本开始，LangChain 和 LangGraph 这两个项目开始融合，**LangGraph正式退居幕后，成为LangChain构建Agent的底层基石**。
- 随着LangGraph成为LangChain构建Agent的底层基石，Agent相关的API也得到了统一

因此LangChain v1.0和LangGraph v1.0 需要相互搭配使用，两者均不适用于 0.x 版本。

> v1.0的LangChain文档删除了对底层 `Runnable`系列抽象接口的介绍，这部分的内容如果想要了解，还得去找GitHub上 LangChain v0.3.x 文档。

**（2）Agent构建引入了Middleware**

新的Agent架构支持Middleware机制，可以更加方便的控制Agent流程中每步的操作

**（3）改进了结构化输出（Structured output）的能力**

`create_agent()`函数提供的结构化输出的生成现在直接在 Agent 的主执行循环中完成，**不再需要额外的 LLM 调用来解析结果**，从而降低了延迟和成本。

**二、统一模型输出**

LangChain的LLM/ChatModel的Response里，引入`.content_blocks`属性：

- 统一了不同模型提供商的消息内容
- 提供了类型提示
- 此属性是延迟加载的，提供了兼容性

**三、简化包结构**

精简了`langchain`包的命名空间，`langchain`包现在聚焦如下几个子module：

| Module                  | What’s available                                 | Notes                                 |
| :---------------------- | :----------------------------------------------- | :------------------------------------ |
| `langchain.messages`    | Message types, `content blocks`, `trim_messages` | **Re-exported** from `langchain-core` |
| `langchain.chat_models` | `init_chat_model`, `BaseChatModel`               | Unified model initialization          |
| `langchain.embeddings`  | `Embeddings`, `init_embeddings`                  | Embedding models                      |
| `langchain.tools`       | `@tool`, `BaseTool`, injection helpers           | **Re-exported** from `langchain-core` |
| **`langchain.agents`**  | `create_agent`, `AgentState`                     | Core agent creation functionality     |

`langchain`包顶层的`__init__.py`里没有引入任何内容，`langchain.agents`模块现在是langchain包的核心。

`langchain-core`包依然存在，并且依旧定义了`langchain`所有抽象组件，包括v0.3版本的`Runnable`接口抽象。

`langchain-community`包也依旧存在，不过一个重要变化是：之前在`langchain`v0.3版本，会直接导入`langchain-community`里的内容，v1.0版本不再直接导出了，需要用户手动显式直接从`langchain-community`里导入。


------
## LangGraph

官方文档：

- [Release -> What's new in LangGraph v1](https://docs.langchain.com/oss/python/releases/langgraph-v1)
- [LangGraph v1 migration guide](https://docs.langchain.com/oss/python/migrate/langgraph-v1)

对于LangGraph来说，从v0.6.11版本升级到 v1.0.0 版本：

- **核心架构和组件基本没有什么变化**
- 提高了执行时的稳定性
- 和LangChain-V1.x无缝对接

最大的改动就是`langgraph.prebuilt`模块里的`create_react_agent()`函数被标记为废弃，由`langchain.agents`模块统一Agent函数`create_agent()`代替了。

不过**由于v1.0的LangChain已经以LangGraph为基石进行了核心组件的重写，官方建议使用LangChain-v1.0提供的API，普通场景下不太需要直接基于LangGraph来构建Agent**。

------
## Deep-Agents

Deep-Agents是 v1.0 新增的包，专门用于构建复杂任务的Agent。


------------------------------------------------------------

# LangChain-Core:v1.0

简单看了下v1.0版本的`langchain-core`模块源码，感觉大体上相比v0.3.x变化不大，核心还是基于`Runnable`接口抽象实现。

包内容如下：
```text
## langchain_core v1.0.1 模块及文件
### 子模块
- api
- callbacks
- document_loaders
- documents
- embeddings
- example_selectors
- indexing
- language_models
- load
- messages
- output_parsers
- outputs
- prompts
- runnables
- tools
- tracers
- utils
- vectorstores

### 根目录文件
- __init__.py
- _import_utils.py
- agents.py
- caches.py
- chat_history.py
- chat_loaders.py
- chat_sessions.py
- env.py
- exceptions.py
- globals.py
- prompt_values.py
- pydantic.py
- rate_limiters.py
- retrievers.py
- stores.py
- structured_query.py
- sys_info.py
- version.py
```

对比 v0.3.x 版本的包结构，可以发现基本没有变化，说明`langchain_core`作为基础包，没有大的变动。


## `messages`模块

`base.py` 里的 `BaseMessage` 基类，新增了一个`content_block`属性，用于统一消息内容：
- 此属性是一个property，会根据不同的模型提供商，尝试对`BaseMessage.content`属性进行解析，返回类型更加安全的对象，也支持多模态模型返回的音视频。
- 返回类型是一个`list[ContentBlock]`。


----------------------------------------------------------------

# LangChain:v1.0

v1.0版本的`langchain`包只有如下模块了。

```text
## langchain v1.0.1 模块及文件
### 子模块
- messages
- chat_models
- tools
- embeddings
- agents
- rate_limiters

### 根目录文件
- __init__.py
- pytyped
```

相比于 **v0.3.80** 版本里，删除了如下模块：
- `langchain.chains`
- `langchain.memory`
- `langchain.callbacks`
- `langchain.document_loaders`
- `langchain.document_transformers`
- `langchain.evaluation`
- `langchain.output_parsers`
- `langchain.prompts`
- `langchain.retrievers`
- `langchain.vectorstores`
- ...

应该说，`langchain` v1.x 的包里，只有 `agents` 和 `tools` 模块有实质性的内容，其他的模块都很简略，大部分是从 langchain-core 中导入。

其实从 v0.3.x 版本开始，`langchain` 包里的一些模块就只是 `langchain_core` 模块里对应包的导入套壳了。

---------------
## `agents`模块 - KEY

这个模块是 LangChain v1.0 改动最大的模块，删除了 v0.6.x 版的许多内容，相当于重构了。

v1.0 里此模块的内容如下：
- `factory.py`: 里面实现了 `create_agent()` 函数用于快速创建 Agent 应用 —— KEY
- `middleware`包: v1.x 版本配合 `create_agent()` 函数使用的中间件，详细介绍见下面。
- `structured_outputs.py`: 配合 `create_agent()` 函数，用于处理Agent的结构化输出结果。

---------------
### `create_agent()` 使用介绍 - KEY

`create_agent()` 函数返回的是 LangGraph 的 `CompiledStateGraph` 对象。

主要参数说明如下：
- `name: str`: Agent名称
- `model: str | BaseChatModel`: 配置使用的模型，可以直接传入配置好的模型对象，也可以传入模型名称（此时会调用`init_chat_model()`方法来初始化模型对象）
- `system_prompt: str`: 系统角色提示词
- `state_schema`: `TypedDict` 类，**用于定义底层 LangGraph 里的 Graph State**。
  - 默认是 `AgentState[ResponseT]`，如果要自定义，则必须继承 `AgentState[ResponseT]`，因为其中定义了几个公共字段（有一个`messages`字段）
  - 这里配置的 `state_schema` 会被用作 base schema，**所有middlewares中定义的`state_schema`字段都会被合并到此base schema中**。
  - 预置的Middleware会用到 `AgentState[ResponseT]` 中定义的 `messages` 字段。
  - Middleware 的各类 hook 方法也会使用这个 state_schema
- `context_schema`: `TypedDict`/`dataclass`/pydantic `BaseModel` **类**（不是对象），用于定义LangGraph的 `Runtime[ContextT]` 里的 `ContextT` 泛型。
  - 这个schema的实例对象可以在调用`CompiledStateGraph.invoke()`方法时作为`context`参数传入，之后可以在LangGraph节点中的`Runtime[ContextT].context`属性获取了。
- `tools`: 工具列表。
  - 查看源码可以发现，**传入的所有tools（包括所有middleware的tools）都被封装到了下面介绍的`tools`模块的 `_ToolNode` 类中**。 
- `middleware`: 中间件配置列表
- `checkpointer: Checkpointer`: LangGraph Checkpointer 对象
- `store: BaseStore`: LangGraph Store 对象
- `response_format`: Structured Output 相关配置，详细介绍见下面的 `structured_outputs.py` 说明。
- `interrupt_before: list[str]`: 通过name指定要在哪些node **执行前** 触发中断
- `interrupt_after: list[str]`: 通过name指定要在哪些node **执行后** 触发中断
- `debug: bool`:
- `cache: BaseCache`:

---------------
### Middleware

Middleware 的相关实现在 `langchain.agents.middleware` 中。

v1.0版新增的middleware功能是**专门配合`create_agent()`函数使用的，LangGraph框架本身并没有提供中间件这一抽象**。

`creage_agent()`函数是基于LangGraph创建的Agent。

`create_agent()`里的middleware调用时机如官方文档图片所示：

<img src="https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=840&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=e9b14e264f68345de08ae76f032c52d4" alt="AgentMiddleware.hook" style="zoom:80%;" align="left"/>

上述调用的hook方法是由`AgentMiddleware`基类定义的（[官方文档：Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)），所有Middleware都要继承此类。

`AgentMiddleware(Generic[StateT, ContextT])` 是一个泛型类，需要`StateT`和`ContextT`两个泛型参数：

- `StateT`: 传递给Middleware的状态对象schema，需要继承自 `AgentState` 类
- `ContextT`: 底层LangGraph的 `Runtime[ContextT]` 泛型参数

`AgentMiddleware(Generic[StateT, ContextT])` 定义了如下两个属性：

- `state_schema: type(StateT)`: 状态对象的类
- `tools: List[BaseTool]`: 该 middleware 所附加的工具列表 —— 这个不知道有什么用。

自定义Middleware时，需要继承`AgentMiddleware`并实现其中的某个方法，可实现的hook方法分为如下两类：

**（1）Node-style hooks**：在固定时机进行调用的hook，一般用于logging、validation、state update等操作。

- agent启动停止：每次调用`invoke()`等方法时执行一次
  - `before_agent`/`abefore_agent`
  - `after_agent`/`aafter_agent`

- model调用：每次发生模型调用的前后，一次`invoke()`方法内部可能有多次模型调用
  - `before_model`/`abefore_model`
  - `after_model`/`aafter_model`

Node-style hook方法的签名为：
- 入参: `state: StateT, runtime: Runtime[ContextT]`
  - `state` 就是该middleware的`state`对象
  - `runtime` 就是LangGraph的`Runtime[ContextT]`对象
- 返回值: `dict[str, Any] | None`

（**2）Wrap-style hooks**：用于**干预agent执行流程**，一般用于retries、caching等操作。

- `wrap_model`/`awrap_model`：每次模型调用前后执行，方法签名为：
  - 入参: 
    - `request: ModelRequest`
    - `handler: Callable[[ModelRequest], ModelResponse]`
  - 返回值: `ModelCallResult`

- `wrap_tool_call`/`awrap_tool_call`，每次工具调用前后执行
  - 入参: 
    - `request: ToolCallRequest`
    - `handler: Callable[[ToolCallRequest], ToolMessage | Command]`
  - 返回值: `ToolMessage | Command`

除了继承`AgentMiddleware`类的方式外，还提供了如下hook装饰器来简化使用：

- `@before_agent`/`@after_agent`
- `@before_model`/`@after_model`
- `@wrap_model_call`
- `@wrap_tool_call`

这些装饰器内部也是将被装饰函数使用动态类定义的技巧封装成`AgentMiddleware`的子类以供使用的。




### Built-in Middlewares

官方文档 [Built-in Middlewares](https://docs.langchain.com/oss/python/langchain/middleware/built-in).

LangChain 默认提供了如下built-in middleware：

#### HumanInTheLoopMiddleware

参考官方文档 [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop).

`HumanInTheLoopMiddleware` 的作用是**在模型调用之后，如果有工具调用，判断哪些工具调用需要触发人工处理**。

它配置了如下两个属性：
- `description_prefix: str`，触发HIL时的描述前缀
- `interrupt_on: dict[str, bool | InterruptOnConfig]`: 一个dict，配置需要中断的tool name
  - key 是需要触发中断的tool name
  - value，该工具的中断配置，可选值如下：
    - `True`, 表示触发中断，后续允许 `approve, edit, and reject` 等HIL结果
    - `False`, 表示自动批准该工具的调用
    - `InterruptOnConfig`对象, 表示其他中断配置：
      - `allowed_decisions`, 允许的HIL结果，可选值为 `Literal["approve", "edit", "reject"]`
      - `description: str`, 触发HIL时的描述

源码实现逻辑大致如下：
- 实现 `AgentMiddleware` 的 `after_model` hook 方法
- 在`after_model` 方法里，从 `state['messages']` 中获取 `AIMessage`
- 从 `AIMessage.tool_calls` 里获取工具调用相关信息，并判断是否在 `self.interrupt_on` 里配置的需要中断的工具
- 如果有需要中断的工具调用，则组织好信息，然后调用 `interrupt()` 方法触发中断。
- `interrupt()` 方法的返回值里，会检查 `decisions` 这个属性， 
  - 因此要求恢复执行时的 `Command` 对象的 `resume` 参数接受的dict里必须要包含 `decisions` 这个 key，
  - `decisions` 是一个 `List[Dict[str, Any]]`，`Dict`的一个key是 `type`，value是 `Literal["approve", "edit", "reject"]`。

#### ModelCallLimitMiddleware


---------------
### 结构化输出

官方文档 [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)

`langchain.agents.structured_outputs.py` 用于处理Agent的输出结果。

Agent捕获到结构化输出之后，会存放在state对象的`'structured_response'`这个Key中。

该文件里定义了如下4种结构化输出处理策略——对应`create_agent()`方法的`response_format`取值，其中`SchemaT`是泛型表示，需要用户自己定义一个数据类，用于封装结构化输出结果。
- `None`: 默认值，不处理输出结果
- `ProviderStrategy[SchemaT]`: 使用**模型提供商（Model Provider）**原生的结构化输出结构，这是**最可靠的方式**，使用起来也比较简单，推荐优先使用。
- `ToolStrategy[SchemaT]`: 采用 **Tool calling 方式**处理结构化输出结果。当模型提供商不支持原生结构化输出时，可以采用此方式，但是不那么稳定。
- `AutoStrategy[SchemaT]`: 自动选择处理策略，此时也可以简单的使用 `type[SchemaT]`

对于 `ToolStrategy[SchemaT]`（它其实是一个`dataclass`类），它定义了如下属性：
- `schema: type[SchemaT]`: 必填属性，存放结构化输出结果的数据类，可以是 `TypedDict`/`dataclass`/pydantic `BaseModel`类。
- `tool_message_content: str`: 自定义文本，**在成功获取结构化结果时，会返回此文本来代替原本的JSON** —— 感觉用处不大
- `handle_errors`: 在获取结构化结果时，如果发生错误，如何处理。有如下的选项：
  - `True`: 默认值，会使用默认的异常信息模板返回错误信息
  - `False`: 不处理错误，将异常向上抛出
  - `str`: 自定义异常信息，一旦抛出结构化结果解析错误，都会返回此文本
  - `type[Exception]` / `tuple[type[Exception],...]`: 只捕获此异常/多个异常，并使用默认异常模板返回异常信息，**并进行重试**；对于其他异常则会继续抛出
  - `Callable[[Exception], str]`: 自定义的异常处理函数，返回处理后的异常信息

---------------
## ~~`tools` 模块~~

> v1.x 版本的 `tools` 模块改动很大，相当于重构了。

此模块里面只有一个 `toos_node.py`，其中定义了如下有用的类：
- `_ToolNode`: 配合`create_agent()` 函数使用，用于对传入的tools列表进行封装
- `ToolCallRequest`:
- `ToolRuntime`:
- `InjectedState`:
- `InjectedStore`:

> **从v1.0.3版本开始，`tools`模块里的内容改为从`langgraph.prebuilt.tool_node`里导入**，没有重新实现。
>
> 因此下面的说明只适用于 v1.0.3 版本之前，v1.0.3版本开始，此模块就是套壳了。

查看`_ToolNode`类的源码（实际上和`langgraph.prebuilt.tool_node`里的`ToolNode`源码逻辑非常相似）可以发现：
- 它对传入的tools列表进行封装时，有一个`__tools_by_name: dict[str, BaseTool]`属性，**存储每个tool的name和tool对象的映射关系**
- 调用`invoke()`方法时，会对传入的`input`进行解析检查，获取其中`AIMessage`对象的`.tool_calls`列表
- 接着调用`_ToolNode._func()`方法，其中会采用 `executor.map(self._run_one, tool_calls, input_types, tool_runtimes))` 方式**并行调用每个Tool**
- `_ToolNode._run_one()`方法中，会先根据tool_call.name从`__tools_by_name`属性中获取对应的tool对象，将调用上下文封装成`ToolCallRequest`对象
- 调用`_ToolNode._execute_tool_sync()` 方法，其中会调用tool的`invoke()`方法，并传入该tool_call的args
- 最后将tool的输出结果封装成`ToolMessage`对象并返回



------

## `agents`模块源码研究

### `create_agent()`函数源码注释

基于 LangChain **v1.2.15** 版本。

```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache[Any] | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
]:
    # init chat model
    if isinstance(model, str):
        model = init_chat_model(model)

    # Convert system_prompt to SystemMessage if needed
    system_message: SystemMessage | None = None
    if system_prompt is not None:
        if isinstance(system_prompt, SystemMessage):
            system_message = system_prompt
        else:
            system_message = SystemMessage(content=system_prompt)

    # Handle tools being None or empty
    if tools is None:
        tools = []

    # Convert response format and setup structured output tools
    # Raw schemas are wrapped in AutoStrategy to preserve auto-detection intent.
    # AutoStrategy is converted to ToolStrategy upfront to calculate tools during agent creation,
    # but may be replaced with ProviderStrategy later based on model capabilities.
    initial_response_format: ToolStrategy[Any] | ProviderStrategy[Any] | AutoStrategy[Any] | None
    if response_format is None:
        initial_response_format = None
    elif isinstance(response_format, (ToolStrategy, ProviderStrategy)):
        # Preserve explicitly requested strategies
        initial_response_format = response_format
    elif isinstance(response_format, AutoStrategy):
        # AutoStrategy provided - preserve it for later auto-detection
        initial_response_format = response_format
    else:
        # Raw schema - wrap in AutoStrategy to enable auto-detection
        initial_response_format = AutoStrategy(schema=response_format)

    # For AutoStrategy, convert to ToolStrategy to setup tools upfront
    # (may be replaced with ProviderStrategy later based on model)
    tool_strategy_for_setup: ToolStrategy[Any] | None = None
    if isinstance(initial_response_format, AutoStrategy):
        tool_strategy_for_setup = ToolStrategy(schema=initial_response_format.schema)
    elif isinstance(initial_response_format, ToolStrategy):
        tool_strategy_for_setup = initial_response_format

    structured_output_tools: dict[str, OutputToolBinding[Any]] = {}
    if tool_strategy_for_setup:
        for response_schema in tool_strategy_for_setup.schema_specs:
            structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
            structured_output_tools[structured_tool_info.tool.name] = structured_tool_info
            
    # ---------------------------------------------------------------------------------
    # 这里收集了所有 middleware 的 tools，注意这使用了两个for循环对每个中间件的tools进行展开
    middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]

    # ---------------------------------------------------------------------------------
    # 接下来收集所有 middleware 的 wrap_tool_call/awrap_tool_call 方法，并封装成串行的wrapper调用函数
    # Collect middleware with wrap_tool_call or awrap_tool_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_wrap_tool_call = [
        m
        for m in middleware
        if m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
        or m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
    ]

    # Chain all wrap_tool_call handlers into a single composed handler
    wrap_tool_call_wrapper = None
    if middleware_w_wrap_tool_call:
        wrappers = [
            # traceable 是 LangSmith 提供的可观测性装饰器函数，不影响原有函数逻辑
            traceable(name=f"{m.name}.wrap_tool_call", process_inputs=_scrub_inputs)(
                m.wrap_tool_call
            )
            for m in middleware_w_wrap_tool_call
        ]
        wrap_tool_call_wrapper = _chain_tool_call_wrappers(wrappers)

    # Collect middleware with awrap_tool_call or wrap_tool_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_awrap_tool_call = [
        m
        for m in middleware
        if m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
        or m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
    ]

    # Chain all awrap_tool_call handlers into a single composed async handler
    awrap_tool_call_wrapper = None
    if middleware_w_awrap_tool_call:
        async_wrappers = [
            traceable(name=f"{m.name}.awrap_tool_call", process_inputs=_scrub_inputs)(
                m.awrap_tool_call
            )
            for m in middleware_w_awrap_tool_call
        ]
        awrap_tool_call_wrapper = _chain_async_tool_call_wrappers(async_wrappers)

    # ---------------------------------------------------------------------------------
    # 配置 tools，这里区分的 内置工具 和 用户工具函数
    # Setup tools
    tool_node: ToolNode | None = None
    # Extract built-in provider tools (dict format) and regular tools (BaseTool/callables)
    built_in_tools = [t for t in tools if isinstance(t, dict)]
    regular_tools = [t for t in tools if not isinstance(t, dict)]

    # Tools that require client-side execution (must be in ToolNode)
    available_tools = middleware_tools + regular_tools

    # ---------------------------------------------------------------------------------
    # ToolNode 中只封装了 中间件tools 和 regular_tools
    # Create ToolNode if we have client-side tools OR if middleware defines wrap_tool_call
    # (which may handle dynamically registered tools)
    tool_node = (
        ToolNode(
            tools=available_tools,
            wrap_tool_call=wrap_tool_call_wrapper,
            awrap_tool_call=awrap_tool_call_wrapper,
        )
        if available_tools or wrap_tool_call_wrapper or awrap_tool_call_wrapper
        else None
    )

    # Default tools for ModelRequest initialization
    # Use converted BaseTool instances from ToolNode (not raw callables)
    # Include built-ins and converted tools (can be changed dynamically by middleware)
    # Structured tools are NOT included - they're added dynamically based on response_format
    if tool_node:
        default_tools = list(tool_node.tools_by_name.values()) + built_in_tools
    else:
        default_tools = list(built_in_tools)

    # ---------------------------------------------------------------------------------
    # 检查所有中间件的 before_agent / before_model / after_model / after_agent 方法是自己实现的
    # validate middleware
    if len({m.name for m in middleware}) != len(middleware):
        msg = "Please remove duplicate middleware instances."
        raise AssertionError(msg)
    middleware_w_before_agent = [
        m
        for m in middleware
        if m.__class__.before_agent is not AgentMiddleware.before_agent
        or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
    ]
    middleware_w_before_model = [
        m
        for m in middleware
        if m.__class__.before_model is not AgentMiddleware.before_model
        or m.__class__.abefore_model is not AgentMiddleware.abefore_model
    ]
    middleware_w_after_model = [
        m
        for m in middleware
        if m.__class__.after_model is not AgentMiddleware.after_model
        or m.__class__.aafter_model is not AgentMiddleware.aafter_model
    ]
    middleware_w_after_agent = [
        m
        for m in middleware
        if m.__class__.after_agent is not AgentMiddleware.after_agent
        or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
    ]
    
    # ---------------------------------------------------------------------------------
    # 对所有中间件的 wrap_model_call / awrap_model_call 方法进行 校验 和 chain 封装
    # Collect middleware with wrap_model_call or awrap_model_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_wrap_model_call = [
        m
        for m in middleware
        if m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
        or m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
    ]
    # Collect middleware with awrap_model_call or wrap_model_call hooks
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # when middleware doesn't support the execution path
    middleware_w_awrap_model_call = [
        m
        for m in middleware
        if m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
        or m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
    ]

    # Compose wrap_model_call handlers into a single middleware stack (sync)
    wrap_model_call_handler = None
    if middleware_w_wrap_model_call:
        sync_handlers = [
            traceable(name=f"{m.name}.wrap_model_call", process_inputs=_scrub_inputs)(
                m.wrap_model_call
            )
            for m in middleware_w_wrap_model_call
        ]
        wrap_model_call_handler = _chain_model_call_handlers(sync_handlers)

    # Compose awrap_model_call handlers into a single middleware stack (async)
    awrap_model_call_handler = None
    if middleware_w_awrap_model_call:
        async_handlers = [
            traceable(name=f"{m.name}.awrap_model_call", process_inputs=_scrub_inputs)(
                m.awrap_model_call
            )
            for m in middleware_w_awrap_model_call
        ]
        awrap_model_call_handler = _chain_async_model_call_handlers(async_handlers)

    # ---------------------------------------------------------------------------------
    # 去重合并所有中间件的 state_schema
    state_schemas: set[type] = {m.state_schema for m in middleware}
    # Use provided state_schema if available, otherwise use base AgentState
    base_state = state_schema if state_schema is not None else AgentState
    state_schemas.add(base_state)

    # 此方法会对所有中间的state_schema的字段进行合并，返回 3 个 TypeDict，对应3类 schema
    resolved_state_schema, input_schema, output_schema = _resolve_schemas(state_schemas)

    # create graph, add nodes
    graph: StateGraph[
        AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
    ] = StateGraph(
        state_schema=resolved_state_schema,
        input_schema=input_schema,
        output_schema=output_schema,
        context_schema=context_schema,
    )

    # ---------------------------------------------------------------------------------
    # 以下 4 个为闭包函数
    
    # 处理模型的结构化输出
    def _handle_model_output(
        output: AIMessage, effective_response_format: ResponseFormat[Any] | None
    ) -> dict[str, Any]:
        # 使用模型厂商的原生结构化输出结果
        # Handle structured output with provider strategy
        if isinstance(effective_response_format, ProviderStrategy):
            if not output.tool_calls:
                provider_strategy_binding = ProviderStrategyBinding.from_schema_spec(
                    effective_response_format.schema_spec
                )
                try:
                    structured_response = provider_strategy_binding.parse(output)
                except Exception as exc:
                    schema_name = getattr(
                        effective_response_format.schema_spec.schema, "__name__", "response_format"
                    )
                    validation_error = StructuredOutputValidationError(schema_name, exc, output)
                    raise validation_error from exc
                else:
                    return {"messages": [output], "structured_response": structured_response}
            return {"messages": [output]}
        # 使用 tool call 方式获取结构化输出
        # Handle structured output with tool strategy
        if (
            isinstance(effective_response_format, ToolStrategy)
            and isinstance(output, AIMessage)
            and output.tool_calls
        ):
            structured_tool_calls = [
                tc for tc in output.tool_calls if tc["name"] in structured_output_tools
            ]

            if structured_tool_calls:
                exception: StructuredOutputError | None = None
                if len(structured_tool_calls) > 1:
                    # Handle multiple structured outputs error
                    tool_names = [tc["name"] for tc in structured_tool_calls]
                    exception = MultipleStructuredOutputsError(tool_names, output)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, effective_response_format
                    )
                    if not should_retry:
                        raise exception

                    # Add error messages and retry
                    tool_messages = [
                        ToolMessage(
                            content=error_message,
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                        for tc in structured_tool_calls
                    ]
                    return {"messages": [output, *tool_messages]}

                # Handle single structured output
                tool_call = structured_tool_calls[0]
                try:
                    structured_tool_binding = structured_output_tools[tool_call["name"]]
                    structured_response = structured_tool_binding.parse(tool_call["args"])

                    tool_message_content = (
                        effective_response_format.tool_message_content
                        or f"Returning structured response: {structured_response}"
                    )

                    return {
                        "messages": [
                            output,
                            ToolMessage(
                                content=tool_message_content,
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            ),
                        ],
                        "structured_response": structured_response,
                    }
                except Exception as exc:
                    exception = StructuredOutputValidationError(tool_call["name"], exc, output)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, effective_response_format
                    )
                    if not should_retry:
                        raise exception from exc

                    return {
                        "messages": [
                            output,
                            ToolMessage(
                                content=error_message,
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            ),
                        ],
                    }

        return {"messages": [output]}

    # 下面的函数只是为了处理模型的结构化输出，对模型bind了用于处理结构化输出的tool，
    # 不会bind用户提供的或者是中间件的 tools
    def _get_bound_model(
        request: ModelRequest[ContextT],
    ) -> tuple[Runnable[Any, Any], ResponseFormat[Any] | None]:
        # Validate ONLY client-side tools that need to exist in tool_node
        # Skip validation when wrap_tool_call is defined, as middleware may handle
        # dynamic tools that are added at runtime via wrap_model_call
        has_wrap_tool_call = wrap_tool_call_wrapper or awrap_tool_call_wrapper

        # Build map of available client-side tools from the ToolNode
        # (which has already converted callables)
        available_tools_by_name = {}
        if tool_node:
            available_tools_by_name = tool_node.tools_by_name.copy()

        # Check if any requested tools are unknown CLIENT-SIDE tools
        # Only validate if wrap_tool_call is NOT defined (no dynamic tool handling)
        if not has_wrap_tool_call:
            unknown_tool_names = []
            for t in request.tools:
                # Only validate BaseTool instances (skip built-in dict tools)
                if isinstance(t, dict):
                    continue
                if isinstance(t, BaseTool) and t.name not in available_tools_by_name:
                    unknown_tool_names.append(t.name)

            if unknown_tool_names:
                available_tool_names = sorted(available_tools_by_name.keys())
                msg = DYNAMIC_TOOL_ERROR_TEMPLATE.format(
                    unknown_tool_names=unknown_tool_names,
                    available_tool_names=available_tool_names,
                )
                raise ValueError(msg)

        # Normalize raw schemas to AutoStrategy
        # (handles middleware override with raw Pydantic classes)
        response_format: ResponseFormat[Any] | Any | None = request.response_format
        if response_format is not None and not isinstance(
            response_format, (AutoStrategy, ToolStrategy, ProviderStrategy)
        ):
            response_format = AutoStrategy(schema=response_format)

        # Determine effective response format (auto-detect if needed)
        effective_response_format: ResponseFormat[Any] | None
        if isinstance(response_format, AutoStrategy):
            # User provided raw schema via AutoStrategy - auto-detect best strategy based on model
            if _supports_provider_strategy(request.model, tools=request.tools):
                # Model supports provider strategy - use it
                effective_response_format = ProviderStrategy(schema=response_format.schema)
            elif response_format is initial_response_format and tool_strategy_for_setup is not None:
                # Model doesn't support provider strategy - use ToolStrategy
                # Reuse the strategy from setup if possible to preserve tool names
                effective_response_format = tool_strategy_for_setup
            else:
                effective_response_format = ToolStrategy(schema=response_format.schema)
        else:
            # User explicitly specified a strategy - preserve it
            effective_response_format = response_format

        # Build final tools list including structured output tools
        # request.tools now only contains BaseTool instances (converted from callables)
        # and dicts (built-ins)
        final_tools = list(request.tools)
        if isinstance(effective_response_format, ToolStrategy):
            # Add structured output tools to final tools list
            structured_tools = [info.tool for info in structured_output_tools.values()]
            final_tools.extend(structured_tools)

        # Bind model based on effective response format
        if isinstance(effective_response_format, ProviderStrategy):
            # (Backward compatibility) Use OpenAI format structured output
            kwargs = effective_response_format.to_model_kwargs()
            return (
                request.model.bind_tools(
                    final_tools, strict=True, **kwargs, **request.model_settings
                ),
                effective_response_format,
            )

        if isinstance(effective_response_format, ToolStrategy):
            # Current implementation requires that tools used for structured output
            # have to be declared upfront when creating the agent as part of the
            # response format. Middleware is allowed to change the response format
            # to a subset of the original structured tools when using ToolStrategy,
            # but not to add new structured tools that weren't declared upfront.
            # Compute output binding
            for tc in effective_response_format.schema_specs:
                if tc.name not in structured_output_tools:
                    msg = (
                        f"ToolStrategy specifies tool '{tc.name}' "
                        "which wasn't declared in the original "
                        "response format when creating the agent."
                    )
                    raise ValueError(msg)

            # Force tool use if we have structured output tools
            tool_choice = "any" if structured_output_tools else request.tool_choice
            return (
                request.model.bind_tools(
                    final_tools, tool_choice=tool_choice, **request.model_settings
                ),
                effective_response_format,
            )

        # No structured output - standard model binding
        if final_tools:
            return (
                request.model.bind_tools(
                    final_tools, tool_choice=request.tool_choice, **request.model_settings
                ),
                None,
            )
        return request.model.bind(**request.model_settings), None

    # 同步调用模型封装，它会调用上面的  _get_bound_model() + _handle_model_output()
    def _execute_model_sync(request: ModelRequest[ContextT]) -> ModelResponse:
        """Execute model and return response.

        This is the core model execution logic wrapped by `wrap_model_call` handlers.

        Raises any exceptions that occur during model invocation.
        """
        # Get the bound model (with auto-detection if needed)
        model_, effective_response_format = _get_bound_model(request)
        messages = request.messages
        if request.system_message:
            messages = [request.system_message, *messages]

        output = model_.invoke(messages)
        if name:
            output.name = name

        # Handle model output to get messages and structured_response
        handled_output = _handle_model_output(output, effective_response_format)
        messages_list = handled_output["messages"]
        structured_response = handled_output.get("structured_response")

        return ModelResponse(
            result=messages_list,
            structured_response=structured_response,
        )

    # 同步调用模型Node，它会调用 _execute_model_sync
    # 然后调用 wrap_model_call_handler，并从response解析生成 Command
    def model_node(state: AgentState[Any], runtime: Runtime[ContextT]) -> list[Command[Any]]:
        """Sync model request handler with sequential middleware processing."""
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_message=system_message,
            response_format=initial_response_format,
            messages=state["messages"],
            tool_choice=None,
            state=state,
            runtime=runtime,
        )

        if wrap_model_call_handler is None:
            model_response = _execute_model_sync(request)
            return _build_commands(model_response)

        result = wrap_model_call_handler(request, _execute_model_sync)
        return _build_commands(result.model_response, result.commands)

    # 异步调用模型封装，它也会调用上面的  _get_bound_model() + _handle_model_output()
    async def _execute_model_async(request: ModelRequest[ContextT]) -> ModelResponse:
        """Execute model asynchronously and return response.

        This is the core async model execution logic wrapped by `wrap_model_call`
        handlers.

        Raises any exceptions that occur during model invocation.
        """
        # Get the bound model (with auto-detection if needed)
        model_, effective_response_format = _get_bound_model(request)
        messages = request.messages
        if request.system_message:
            messages = [request.system_message, *messages]

        output = await model_.ainvoke(messages)
        if name:
            output.name = name

        # Handle model output to get messages and structured_response
        handled_output = _handle_model_output(output, effective_response_format)
        messages_list = handled_output["messages"]
        structured_response = handled_output.get("structured_response")

        return ModelResponse(
            result=messages_list,
            structured_response=structured_response,
        )

    # 异步调用模型Node，它会调用 _execute_model_async
    # 然后调用 awrap_model_call_handler，并从response解析生成 Command
    async def amodel_node(state: AgentState[Any], runtime: Runtime[ContextT]) -> list[Command[Any]]:
        """Async model request handler with sequential middleware processing."""
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_message=system_message,
            response_format=initial_response_format,
            messages=state["messages"],
            tool_choice=None,
            state=state,
            runtime=runtime,
        )

        if awrap_model_call_handler is None:
            model_response = await _execute_model_async(request)
            return _build_commands(model_response)

        result = await awrap_model_call_handler(request, _execute_model_async)
        return _build_commands(result.model_response, result.commands)

    # 添加模型（异步/同步）调用节点
    # Use sync or async based on model capabilities
    graph.add_node("model", RunnableCallable(model_node, amodel_node, trace=False))

    # ---------------------------------------------------------------------------------
    # 所有的 tools，都被封装到一个 Node里
    # Only add tools node if we have tools
    if tool_node is not None:
        graph.add_node("tools", tool_node)

    # ---------------------------------------------------------------------------------
    # 遍历中间件，将所有中间件的 before_agent / before_model / after_model / after_agent 都分别添加为单独的Node
    # Add middleware nodes
    for m in middleware:
        if (
            m.__class__.before_agent is not AgentMiddleware.before_agent
            or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_before_agent = (
                m.before_agent
                if m.__class__.before_agent is not AgentMiddleware.before_agent
                else None
            )
            async_before_agent = (
                m.abefore_agent
                if m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
                else None
            )
            before_agent_node = RunnableCallable(sync_before_agent, async_before_agent, trace=False)
            graph.add_node(
                f"{m.name}.before_agent", before_agent_node, input_schema=resolved_state_schema
            )

        if (
            m.__class__.before_model is not AgentMiddleware.before_model
            or m.__class__.abefore_model is not AgentMiddleware.abefore_model
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_before = (
                m.before_model
                if m.__class__.before_model is not AgentMiddleware.before_model
                else None
            )
            async_before = (
                m.abefore_model
                if m.__class__.abefore_model is not AgentMiddleware.abefore_model
                else None
            )
            before_node = RunnableCallable(sync_before, async_before, trace=False)
            graph.add_node(
                f"{m.name}.before_model", before_node, input_schema=resolved_state_schema
            )

        if (
            m.__class__.after_model is not AgentMiddleware.after_model
            or m.__class__.aafter_model is not AgentMiddleware.aafter_model
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_after = (
                m.after_model
                if m.__class__.after_model is not AgentMiddleware.after_model
                else None
            )
            async_after = (
                m.aafter_model
                if m.__class__.aafter_model is not AgentMiddleware.aafter_model
                else None
            )
            after_node = RunnableCallable(sync_after, async_after, trace=False)
            graph.add_node(f"{m.name}.after_model", after_node, input_schema=resolved_state_schema)

        if (
            m.__class__.after_agent is not AgentMiddleware.after_agent
            or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
        ):
            # Use RunnableCallable to support both sync and async
            # Pass None for sync if not overridden to avoid signature conflicts
            sync_after_agent = (
                m.after_agent
                if m.__class__.after_agent is not AgentMiddleware.after_agent
                else None
            )
            async_after_agent = (
                m.aafter_agent
                if m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
                else None
            )
            after_agent_node = RunnableCallable(sync_after_agent, async_after_agent, trace=False)
            graph.add_node(
                f"{m.name}.after_agent", after_agent_node, input_schema=resolved_state_schema
            )

    # ---------------------------------------------------------------------------------
    # 确定Graph入口Node，优先级为：第1个before_agent -> 第1个before_model -> model
    # Determine the entry node (runs once at start): before_agent -> before_model -> model
    if middleware_w_before_agent:
        entry_node = f"{middleware_w_before_agent[0].name}.before_agent"
    elif middleware_w_before_model:
        entry_node = f"{middleware_w_before_model[0].name}.before_model"
    else:
        entry_node = "model"

    # ---------------------------------------------------------------------------------
    # 确定循环调用的开始节点
    # Determine the loop entry node (beginning of agent loop, excludes before_agent)
    # This is where tools will loop back to for the next iteration
    if middleware_w_before_model:
        loop_entry_node = f"{middleware_w_before_model[0].name}.before_model"
    else:
        loop_entry_node = "model"

    # ---------------------------------------------------------------------------------
    # 确定循环调用的退出节点
    # Determine the loop exit node (end of each iteration, can run multiple times)
    # This is after_model or model, but NOT after_agent
    if middleware_w_after_model:
        loop_exit_node = f"{middleware_w_after_model[0].name}.after_model"
    else:
        loop_exit_node = "model"

    # ---------------------------------------------------------------------------------
    # 确定Graph的终止节点
    # Determine the exit node (runs once at end): after_agent or END
    if middleware_w_after_agent:
        exit_node = f"{middleware_w_after_agent[-1].name}.after_agent"
    else:
        exit_node = END

    graph.add_edge(START, entry_node)
    
    # ---------------------------------------------------------------------------------
    # 最重要的部分：构建Agent循环中的工具调用和模型调用之间的条件边
    # add conditional edges only if tools exist
    if tool_node is not None:
        # Only include exit_node in destinations if any tool has return_direct=True
        # or if there are structured output tools
        tools_to_model_destinations = [loop_entry_node]
        if (
            any(tool.return_direct for tool in tool_node.tools_by_name.values())
            or structured_output_tools
        ):
            tools_to_model_destinations.append(exit_node)

        graph.add_conditional_edges(
            "tools",
            RunnableCallable(
                _make_tools_to_model_edge(
                    tool_node=tool_node,
                    model_destination=loop_entry_node,
                    structured_output_tools=structured_output_tools,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            tools_to_model_destinations,
        )

        # base destinations are tools and exit_node
        # we add the loop_entry node to edge destinations if:
        # - there is an after model hook(s) -- allows jump_to to model
        #   potentially artificially injected tool messages, ex HITL
        # - there is a response format -- to allow for jumping to model to handle
        #   regenerating structured output tool calls
        model_to_tools_destinations = ["tools", exit_node]
        if response_format or loop_exit_node != "model":
            model_to_tools_destinations.append(loop_entry_node)

        graph.add_conditional_edges(
            loop_exit_node,
            RunnableCallable(
                _make_model_to_tools_edge(
                    model_destination=loop_entry_node,
                    structured_output_tools=structured_output_tools,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            model_to_tools_destinations,
        )
    elif len(structured_output_tools) > 0:
        graph.add_conditional_edges(
            loop_exit_node,
            RunnableCallable(
                _make_model_to_model_edge(
                    model_destination=loop_entry_node,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            [loop_entry_node, exit_node],
        )
    elif loop_exit_node == "model":
        # If no tools and no after_model, go directly to exit_node
        graph.add_edge(loop_exit_node, exit_node)
    # No tools but we have after_model - connect after_model to exit_node
    else:
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_after_model[0].name}.after_model",
            default_destination=exit_node,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_after_model[0], "after_model"),
        )

    # ---------------------------------------------------------------------------------
    # 构建所有中间件内部依次调用 before_agent / before_model / after_model / after_agent 之间的顺序边
    # Add before_agent middleware edges
    if middleware_w_before_agent:
        for m1, m2 in itertools.pairwise(middleware_w_before_agent):
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.before_agent",
                default_destination=f"{m2.name}.before_agent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "before_agent"),
            )
        # Connect last before_agent to loop_entry_node (before_model or model)
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_before_agent[-1].name}.before_agent",
            default_destination=loop_entry_node,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_before_agent[-1], "before_agent"),
        )

    # Add before_model middleware edges
    if middleware_w_before_model:
        for m1, m2 in itertools.pairwise(middleware_w_before_model):
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.before_model",
                default_destination=f"{m2.name}.before_model",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "before_model"),
            )
        # Go directly to model after the last before_model
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_before_model[-1].name}.before_model",
            default_destination="model",
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_before_model[-1], "before_model"),
        )

    # Add after_model middleware edges
    if middleware_w_after_model:
        graph.add_edge("model", f"{middleware_w_after_model[-1].name}.after_model")
        for idx in range(len(middleware_w_after_model) - 1, 0, -1):
            m1 = middleware_w_after_model[idx]
            m2 = middleware_w_after_model[idx - 1]
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.after_model",
                default_destination=f"{m2.name}.after_model",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "after_model"),
            )
        # Note: Connection from after_model to after_agent/END is handled above
        # in the conditional edges section

    # Add after_agent middleware edges
    if middleware_w_after_agent:
        # Chain after_agent middleware (runs once at the very end, before END)
        for idx in range(len(middleware_w_after_agent) - 1, 0, -1):
            m1 = middleware_w_after_agent[idx]
            m2 = middleware_w_after_agent[idx - 1]
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.after_agent",
                default_destination=f"{m2.name}.after_agent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "after_agent"),
            )

        # Connect the last after_agent to END
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_after_agent[0].name}.after_agent",
            default_destination=END,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_after_agent[0], "after_agent"),
        )

    # 配置完毕，Compile Graph
    # Set recursion limit to 9_999
    # https://github.com/langchain-ai/langgraph/issues/7313
    config: RunnableConfig = {"recursion_limit": 9_999}
    config["metadata"] = {"ls_integration": "langchain_create_agent"}
    if name:
        config["metadata"]["lc_agent_name"] = name

    return graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config(config)
```



### 源码逻辑重点说明

#### 中间件封装

研究`create_agent()`函数源码可以发现，Middleware的使用有如下几个要注意的地方：

- 所有middleware的`.tools`会被合并到一起，封装到`ToolNode`里
- 所有middleware的`wrap_tool_call`/`awrap_tool_call`方法会被合并成一条链，封装到`ToolNode`里，在每次调用tool前后执行。
- 所有middleware的`wrap_model_call`方法也会被合并成一条链，封装到一个`model_node`（对应源码里的`model_node()`闭包函数）里，在每次模型调用前后执行。
- 所有middleware的`before_agent`/`after_agent`/`before_model`/`after_model`方法，**各自会被封装成LangGraph里的一个Node**，并依次添加之间的边，在指定时机/条件下执行
- 同一个middleware不能多次使用，否则会抛异常，提醒有重复的middleware

> `create_agent()` 函数内部对所有middleware的 `wrap_tool_call` / `wrap_model_call` 进行合并的
> `_chain_model_call_handlers()` / `_chain_tool_call_wrappers()` 方法值得看看（涉及到装饰器和绑定方法的使用）。

最重要的原则如下：

> 每个middleware继承`AgentMiddleware`时，**最好只实现其中一个hook方法**——尽量遵守**单一职责**的实践；
>
> 即使要实现多个hook方法，这些方法之间**不要有关联或者访问共享变量的操作**，因为根据源码里的逻辑，这些方法都是被拆分开使用的，
> `AgentMiddleware`抽象类及其子类只不过是一个封装方法的容器而已，这也要求`AgentMiddleware`子类里最好不要定义实例属性存放共享状态。

#### Tools封装

`create_agent()`使用 `_ToolNode`类（v1.0.3版本开始改为LangGraph的`ToolNode`）封装所有的tools，然后还需要构建 `_ToolNode` 的进入边和输出边：

- 输出边是通过 `factory.py` 里的 `_make_tools_to_model_edge()` 方法构建的：
  - source 是 `_ToolNode`，默认名称为 `tools`
  - `_make_tools_to_model_edge()` 作为 `Graph.add_conditional_edge()` 方法的 `path` 参数
  - 大致逻辑是：大多数情况下，`_ToolNode` 的输出边会指向 `model` 节点；如果**所有**的tool都设置了`return_direct=True`，就指向结束节点
- 输入边是通过 `factory.py` 里 `_make_model_to_tools_edge()` 方法构建的：
  - source 一般是 `model` 节点
  - `_make_model_to_tools_edge()` 作为 `Graph.add_conditional_edge()` 方法的 `path` 参数
  - 大致逻辑是：从 `state['messages']` 的末尾寻找 AIMessage 对象，如果对象的tool_calls属性有值，则遍历并封装tool_call对象，
    并使用LangGraph的 `Send` 对象封装每个 tool_call 信息，返回一个 `List[Send]`。






---------------------------------------------------

# LangChain-Core:v0.3

以下是对 LangChain v0.3版本 的各个package进行简单总结。

> LangChain v0.3 版本的文档现在只能在Github历史提交记录里看到了：https://github.com/langchain-ai/langchain/tree/v0.3/docs/docs。

package名称为`langchain_core`，需要关注的有如下内容。  

大部分模块的说明可以在该模块的 `__init__.py` 文件中找到。

`langchain_core` v0.3.80 版本的源码内容如下：

```text
## langchain_core 模块及文件
### 子模块
- api
- beta
- callbacks
- document_loaders
- documents
- embeddings
- example_selectors
- indexing
- language_models
- load
- messages
- output_parsers
- outputs
- prompts
- pydantic_v1
- runnables
- tools
- tracers
- utils
- vectorstores

### 根目录文件
- __init__.py
- _import_utils.py
- agents.py
- caches.py
- chat_history.py
- chat_loaders.py
- chat_sessions.py
- env.py
- exceptions.py
- globals.py
- memory.py
- prompt_values.py
- pydantic.py
- rate_limiters.py
- retrievers.py
- stores.py
- structured_query.py
- sys_info.py
- version.py
```

---------------------------------------------------
## Chain基础

这部分的内容是LangChain里的基础，主要用于 Chain 的构建，并支持 LangChain Expression Language (LCEL) 语法。


### `runnables`模块 - KEY

这个模块是langchain_core模块的核心模块，基于 Runnable设计模式 和 *LangChain Expression Language (LCEL)* 定义了一系列的接口规范。
也是实现 Chain 的核心模块。 

这里重点介绍如下文件里定义的一些常用抽象基类。

#### `base.py`

##### `Runnable`

它是LangChain里大部分对象执行的基本单元对象，是LangChain里的核心抽象基类，详细介绍可以参考官方文档[Conceptual Guide -> Runnable interface](https://python.langchain.com/docs/concepts/runnables/).

它重载了运算符`|`（重写了`__or__`/`__oro__`方法），并提供了`pipe`方法，为LCEL的 `|` 语法提供了支持。

`class Runnable(ABC, Generic[Input, Output])` 是抽象类，同时也是泛型类。

一、重要属性

`Runnable`只定义了一个`name`属性，用于标识Runnable对象的名称。

但是定义了如下几个Property：

- `InputType`，对应泛型参数的`Input`类
- `OutputType`，对应泛型参数的`Output`类
- `input_schema`，`type[BaseModel]`类
- `output_schema`，`type[BaseModel]`类
- `config_specs`

二、接口方法

`Runnable`定义了如下常用的接口方法:

- `invoke`/`ainvoke`: 输入单条，输出结果。这里的 `invoke` 方法被标记为抽象方法，所以继承 `Runnable` 时，必须实现 `invoke` 方法。
- `batch`/`abatch`: 批量invoke，输出结果
- `stream`/`astream`: 流式方法，内部会`yield self.invoke()`，所以具体的流式输出逻辑还需要`invoke()`方法的实现支持。
- `batch_as_completed`/`abatch_as_completed`: 批量invoke直到完成
- `transform`/`atransform`: 用于将输入转换成输出，底层默认是调用`stream`/`astream`方法

> 上述所有方法中，只有 `invoke` 方法是抽象方法，其他方法都有默认实现，所以如果要继承 `Runnable` 时，必须要实现的方法只有 `invoke`。

此外，`Runnable`还定义了如下几个接口方法，它们均返回`RunnableBinding`对象，对当前Runnable对象进行一些封装并附加一些参数/属性：
- `bind(self, **kwargs: Any) -> Runnable[Input, Output]`: 以关键字参数附加一些参数/属性
- `with_config`: 以`RunnableConfig` + 关键字参数附加信息
- `with_listeners`/`with_alisteners`: 给Runnable对象，添加一些监听器，在运行开始，运行完成时，运行出错后，调用对应的监听回调函数。
- `with_types`:
- `with_retry`:
- `with_fallbacks`:
- `as_tool`:

上面的`with_listeners`/`with_alisteners`方法接受的Callable对象签名是：`Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]`


##### `RunnableSerializable`

`class RunnableSerializable(Serializable, Runnable[Input, Output])`是大部分LLM/ChatLLM的基类，**注意，它不是抽象类，不过一般不会直接使用**。

其中的`class Serializable(BaseModel, ABC)`是`load`模块里的类。

`RunnableSerializable`定义了如下两个在运行修改配置的接口方法：
- `configurable_fields`: 
- `configurable_alternatives`: 

##### `RunnableLambda` - KEY

`class RunnableLambda(Runnable[Input, Output])`，用于将任意`Callable`对象封装成`Runnable`对象，很常用。

可以对异步或非异步函数进行封装，但**不适合以stream方式返回的函数** —— 这种情况应该使用 `RunnableGenerator`。

##### `RunnableGenerator`

`class RunnableGenerator(Runnable[Input, Output])`，用于将任意`Generator`对象封装成`Runnable`对象，**适合以stream方式返回的函数**。

##### `RunnableBinding` - KEY

`class RunnableBinding(RunnableBindingBase[Input, Output])`，用于对`Runnable`对象进行封装并附加一些参数/属性，并返回一个`RunnableBinding`对象 —— 对原有的`Runnable`对象进行了包装。

它相当于一个 **Runnable 装饰器**，LangChain框架内部很多地方都用到了它。

它继承自`class RunnableBindingBase(RunnableSerializable[Input, Output])`类，该类定义了如下属性：
- `bound: Runnable[Input, Output]`，内部包装的 `Runnable` 对象
- `config: RunnableConfig`，附加到 `bound` 对象的运行配置

`RunnableBindingBase`类还重写了`invoke`, `batch`, `stream` 等方法，将这些方法的调用附加`config`配置后转发给 `bound` 对象，并返回结果。

`RunnableBinding`类定义了如下几个方法，和`Runnable`里的方法对应。

- `bind()`
- `with_config()`
- `with_listeners()`
- `with_types()`
- `with_retry()`


##### 总结

`Runnable` 和 `RunnableSerializable` 两个类是整个`runnable`模块的基础。

除此之外，`base.py`文件里，还提供了一些Runnable的常用封装类，方便使用，列举如下：
- `RunnableSequence`: 组合多个Runnable对象，LCEL语法的 `|` 运算符返回的就是这个对象，也很常用
- `RunnableParallel`: 用于并行执行多个 Runnable 对象。
  它将输入数据分发给多个独立的处理步骤，并将它们的结果合并为一个输出字典。
- `RunnableEach`:


#### `config.py`和`configurable.py`

`config.py`模块定义了`RunnableConfig`类 —— 它实际上就是一个Dict对象（`TypedDict`），用于封装`Runnable`对象运行时参数。    
默认定义的`Runnable`参数如下：
- run_id: UUID类型
- run_name: str类型，Runnable对象名称
- metadata: dict
- tags: list[str]
- callbacks: `Union[list[BaseCallbackHandler], BaseCallbackManager]`，回调函数/管理器配置
- configurable: dict[str, Any]，这个参数用于**接受自定义的配置**。
- max_concurrency
- recursion_limit

`configurable.py`里提供了如下两个常用类，
配合上面`RunnableSerializable`的`configurable_fields`和`configurable_alternatives`方法使用：

- `RunnableConfigurableFields`
- `RunnableConfigurableAlternatives`


#### `passthrough.py`

定义了如下类：
- `RunnablePassthrough`: 原样返回输入，相当于一个 identity function —— 不知道这有啥用。。。
- `RunnableAssign`: 用于在链式操作中动态地为**输入**数据添加或更新字段，允许你在key-value数据流中插入新的键值对，或者修改现有的键值对，而无需手动编写复杂的适配器函数
- `RunnablePick`


#### `history.py`

只有一个 `RunnableWithMessageHistory` 类——注意，**它不是抽象类**。

它和 `chat_history.py`里的 `BaseChatMessageHistory` 抽象类配合使用，并且支持通过 LCEL 表达式和 LangGraph 集成。

**使用说明**

`RunnableWithMessageHistory`使用时有3个需要关注的概念：

（1）Runnable对象    

`RunnableWithMessageHistory` 是**对一个可运行对象（比如链或模型）的封装**。这个可运行对象可以是：
- 一个简单的语言模型（LLM）。
- 一个复杂的链（chain），例如 ConversationChain。
- 其他实现了 `Runnable` 接口的对象。

（2）消息历史（Message History）    

消息历史通常由 `BaseChatMessageHistory`实现类 管理。它记录了用户与助手之间的交互消息。

（3）动态加载历史    

`RunnableWithMessageHistory` 需要通过一个函数动态加载消息历史——对应于`get_session_history`属性。    
这使得你可以从外部存储（如数据库）中获取历史记录，并在每次运行时动态更新。


`RunnableWithMessageHistory`的初始化参数如下：
- `get_session_history`: 类型是一个`Callable`对象，要求必须返回一个`BaseChatMessageHistory`——也就是一个简单工厂函数。    
  它的作用是**根据不同用户的身份，加载对应的消息历史**，所以要采用简单工厂函数的方式。
- `history_factory_config`: 类型是`Sequence[ConfigurableFieldSpec]`。    
  作用是说明简单工厂函数的参数，**简单工厂函数有多个参数时会用到**，如果简单工厂函数只需要一个参数，则可以省略。
- `history_messages_key`: `Optional[str]`类型，用于指定 prompt 中，填充历史消息的key，默认是None。
- `input_messages_key`: `Optional[str]`类型，用于指定从输入中获取某个消息的key，默认是None。
- `output_messages_key`: `Optional[str]`类型，用于指定从输出中获取某个消息的key，默认是None。

> 如果封装的 Runnable 对象的输入是一个 Dict，那么 `history_messages_key` 和 `input_messages_key`都必须要设置，否则可能获取不了历史消息。


`RunnableWithMessageHistory`的大致执行逻辑如下：
1. 初始化时，构造一个`RunnableSequence`，按顺序封装如下调用：    
   `self._enter_history` -> `RunnablePassthrough.assign` -> `Chain` -> `self._exit_history`

2. 在 `self._enter_history` 里，     
    2.1 从RunnableConfig里获取`BaseChatMessageHistory`对象，读取其中**所有**历史消息；    
    2.2 如果没有设置`history_messages_key`和`input_messages_key`，则直接将所有历史消息作为输入；   
    2.3 如果没有设置`history_messages_key`，但设置了`input_messages_key`，则调用`self._get_input_messages`，
      从输入中获取指定key消息，封装成`HumanMessage`追加到2.1中的历史消息列表里    
    2.4 返回历史消息列表，进入下一个Runnable

3. 只要`history_messages_key`或者`input_messages_key`有一个存在，则使用`RunnablePassthrough.assign`封装 步骤2 中的 Runnable 对象    
    3.1 在 input 中新增一个key，存放步骤2返回的历史消息列表   
    3.2 这个key的名称优先使用 `history_messages_key`，没有则使用 `input_messages_key`   
    如果`history_messages_key`和`input_messages_key`都没有设置，那么就不会在input中新增存放历史消息的key。

> 这一步其实很重要，如果`history_messages_key`和`input_messages_key`都没有设置，不执行`RunnablePassthrough.assign`封装的话，
> 那么首先执行就是步骤2中的`self._enter_history`，但是该方法返回值是 `list[BaseMessage]`；
> 后续的 Chain 本来是期望接收一个 Dict 的，对于 `list[BaseMessage]` 的处理很可能出问题。
> 如果执行了 `RunnablePassthrough.assign` 封装的话，那么返回的肯定是一个 Dict，那么后续的 Chain 就不会出问题。

4. 执行`Chain`

5. `self._exit_history`作为`Chain.with_listeners(on_end= ... )`监听器调用，在Chain执行完时触发：    
    5.1 从RunnableConfig里获取`BaseChatMessageHistory`对象    
    5.2 调用`self._get_input_messages(inputs)`，尝试从input中以`input_messages_key`（没有则使用'input'作为默认key）获取消息，封装为`HumanMessage`
    5.3 获取步骤4中的output，调用`self._get_output_messages(outpus)`，尝试以`output_messages_key`为key从output中获取消息，封装为`AIMessage`
    5.4 向 `BaseChatMessageHistory`对象中追加 [`HumanMessage`, `AIMessage`]

> `input_messages_key`和`output_messages_key`这两个参数的最大作用是在`self._exit_history`中，此时Chain调用结束，
> 需要使用这两个key分别从 input和output中 获取 用户的输入 和 模型的输出，并存入`BaseChatMessageHistory`对象中。
> 如果没有设置或设置的不对，导致没有获取到用户输入和模型的输出，那么就无法将本次对话存入历史记录中，后续对话也就拿不到历史记录。


------
### `callbacks`模块 - KEY

callbacks模块一般是由`BaseLLM`/`BaseChatModel`/`Chain`对象封装，不直接和Runnable基础类配合使用。

module主要内容有：

#### `base.py`

定义了回调函数的 Mixin 类，回调函数通过 callback handler 定义 一系列Mixin类，大致可以分为如下几类：

- `RetrieverManagerMixin`, `LLMManagerMixin`, `ChainManagerMixin`, `ToolManagerMixin`
- `CallbackManagerMixin`
- `RunManagerMixin`   

这些Mixin类分别定义了各种类型事件的调用方法，比如`on_llm_start`/`on_chat_model_start`/`on_chain_start`等。

此外，还定义了一系列的 CallbackHandler：

- `BaseCallbackHandler`: 同步回调函数handler的接口类，继承了上面大部分的 Mixin 类
- `AsyncCallbackHandler`: 异步回调函数handler的接口类
- `BaseCallbackManager`: 回调函数管理器的基础类
  - 它提供了一系列注册、管理 `BaseCallbackHandler`/`AsyncCallbackHandler` 的方法，以列表的形式存放所有的 callbackhandler
  - 它继承了`CallbackManagerMixin`，但是**并没有实现其中的事件方法**，所以应当看做抽象类


#### `manager.py`

实现了一系列回调管理器的类和方法，需要关注的有如下几个：
- `CallbackManager`: 同步callback handler管理器，继承自`BaseCallbackManager`，实现了其中的事件方法，在对应事件方法里依次调用注册的callbackhandler。
- `AsyncCallbackManager`: 异步callback handler管理器，继承自`BaseCallbackManager`
- `handle_event()`函数：具体执行调用回调函数的地方

#### 其他

- `file.py`: 定义了一个`FileCallbackHandler`供使用，继承自`BaseCallbackHandler`，实现了部分事件方法

- `stdout.py`: 定义了一个`StdOutCallbackHandler`供使用，继承自`BaseCallbackHandler`，实现了部分事件方法

- `streaming_stdout.py`: 定义了一个`StreamingStdOutCallbackHandler`供使用，继承自`BaseCallbackHandler`


**使用说明**

- `base.py`中定义了一系列的Mixin类，它们定义了各个组件的事件方法，比如`on_llm_start`/`on_chat_model_start`/`on_chain_start`等方法。

> 注意，这些事件方法一般不建议有返回值，因为Langchain框架似乎没有明确处理这些返回值。

- `BaseCallbackHandler`组合了上述Mixin类，是所有CallbackHandler（包括`AsyncCallbackHandler`）的基类。    
  - 需要注意的是，虽然`BaseCallbackHandler`/`AsyncCallbackHandler`不是抽象类，但所有事件方法都是空的
  - 所以实际使用时，需要继承此类，并实现自己需要的方法。

- `BaseCallbackManager`是回调管理器的基类，它定义并实现了一些基础方法，
不过一般不需要直接使用此类，而是使用子类`CallbackManager`/`AsyncCallbackManager`等

> 特别要注意的是，LangChain里提供的 CallbackManager 是有层级的，有的是在 Chain 级别调用，有的是在 LLM/ChatModel 级别，最细的级别是 Token 级别.
> 比如 `CallbackManager` 就是在 LLM/ChatModel 级别调用的，所以它实现了 `on_llm_start`/`on_chat_model_start`等方法，
> 这些方法会对每一条 Message 调用一次。
> 方法的返回值是 `list[CallbackManagerForLLMRun]`，其中的 `CallbackManagerForLLMRun` 对应于每一条 Message。
> 而`CallbackManagerForLLMRun` 中实现了 `on_llm_new_token`/`on_llm_new_token`等方法，对应的是 Token 级别。

- `CallbackManager`/`AsyncCallbackManager`虽然有初始化方法，不过langchain框架内部一般使用它提供的classmethod `configure` 方法来初始化并返回对应的实例。

- `CallbackManager`/`AsyncCallbackManager`一般**由下面的`BaseLLM`/`BaseChatModel`/`Chain`(langchain模块提供)封装**，
这些对象都有`callbacks`/`callback_manager`属性，对应的就是这里的`BaseCallbackManager`/`AsyncCallbackManager`或者`CallbackHandler`/`AsyncCallbackHandler`对象列表。

- `BaseLLM`/`BaseChatModel`/`Chain`在配置好CallbackManager后，需要自己在合适的时机调用`on_llm_start`/`on_chat_model_start`/`on_chain_start`等方法，
  来触发配置的所有CallbackHandler。

- `BaseLLM`/`BaseChatModel`/`Chain`一般**并不是在初始化时就实例化并配置 CallbackHandler 的**:
  - 而是在`invoke`/`stream`等方法里调用`CallbackManager.configure()`生成最终要使用的 CallbackHandler 实例。
  - 对于初始化时通过 `callbacks` 传入的回调函数，也会在被并入新的 CallbackHandler 实例。
  - 注意这里`CallbackManager.configure()`里一般不会使用初始化时通过`callback_manager`参数传入的配置，这也是为啥这个参数被废弃的原因。



------
### `load`模块

定义了LangChain里有关对象序列化/反序列化相关的内容。

> LangChain的序列化/反序列化主要基于Pydantic的`BaseModel`实现的。

最重要的是 `serialization.py` 源码，提供了如下抽象类：
- `Serializable`: 支持序列化/反序列化的抽象基类，大部分LangChain对象都基于此抽象类做序列化，它本身继承了`BaseModel`。


------
### `tracers`模块


---------------------------------------------------
## Model IO相关

### `language_model`模块

主要有两类：
- LLMs: 生成式模型，对应于 Completion 任务
- ChatModels: 对话模型，对应于 ChatCompletion 任务

提供的类结构为：
- `base.py`
  - `BaseLanguageModel`，所有语言模型的基类，不过一般不需要继承此类。
- `llms.py`
  - `BaseLLM`，所有LLM的基类，定义了使用时的方法
  - `LLM`, 继承自`BaseLLM`，自定义LLM时，应当继承此类。
- `chat_models.py`
  - `BaseChatModel`，所有聊天模型的基类，定义了使用时的方法。
  - `SimpleChatModel`，继承自`BaseChatModel`，**自定义ChatModel时，应当继承此类**。

**使用说明**

一、`class BaseLanguageModel(RunnableSerializable[LanguageModelInput, LanguageModelOutputVar], ABC)`是所有LLM的基类。

它定义了如下常用属性：

- `metadata`:
- `tags`:
- `verbose: bool`: 是否输出详细日志
- `callbacks`: 回调函数设置，它是一个`Union[list[BaseCallbackHandler], BaseCallbackManager]`，既可以是回调函数列表，也可以是回调管理器。
- `custom_get_token_ids`:

二、`BaseLLM`/`BaseChatModel`继承自`BaseLanguageModel`，它新增了如下属性：

- `callback_manager`: `BaseCallbackManager`类型，回调管理器。

  不过**这个属性和`BaseLanguageModel`的`callbacks`属性功能重复了，所以被标识为废弃的，建议使用`callback_manager`属性**。

三、使用接口

通常使用时，需要关注的是 `BaseLLM` 和 `BastChatModel` 提供的一些方法，列举如下：

- `invoke`/`ainvoke`: 输入单条，输出结果。
- `stream`/`astream`: 流式调用invoke。
- `batch`/`abatch`: 批量invoke，输出结果。

> 以上方法由`langchain_core/runnables/base.py`的`Runnable`抽象类定义。

- `generate_prompt`/`agenerate_prompt`: 输入一批prompt，调用模型产生输出，一般不需要手动调用。
- `predict`/`apredict`: 输入**单条** raw text，调用模型，并以raw text返回结果。
- `predict_messages`/`apredict_messages`: 输入`List[BaseMessage]`，调用模型，并以`BaseMessage`返回结果。

> 以上方法由 `BaseLanguageModel` 定义。

- `generate`/`agenerate`: 调用模型产生输出。由`BaseLLM`/`BaseChatModel`类实现的方法，比较底层，一般不需要手动调用。

> `BaseLLM` 和 `BastChatModel` 也提供了 `__call__` 方法，支持Callable调用的方式，不过看源码里，这种Callable调用被标记为废弃，后续1.0版本可能会移除掉。

整个抽象层次的调用逻辑为：`Runnable`定义抽象方法 --调用--> `BaseLanguageModel`定义抽象方法 --调用--> `BaseLLM`/`BaseChatModel`实现方法。

因此在**实际使用过程中，需要关注的是：`invoke`/`ainvoke`、`batch`/`abatch`、`stream`/`astream` 这3对方法**。

如果想基于`BaseLLM`/`BastChatModel`实现自己的模型，或者想看具体模型的实现，需要重点关注的是模型实现类里的如下方法：
- `_llm_type`：property，用户返回模型的唯一标识，必须要实现
- `_generate`/`_agenerate`: 必须要实现的模型调用方法
- `_stream`/`_astream`: 可选方法

四、结构化输出

`BastChatModel`类还定义了如下两个抽象方法（`BaseLLM`没有）：

（1）`bind_tools`，封装工具调用。

（2）`with_structured_output`，处理结构化输出。


### `messages`模块

用于封装 prompts 和 chat conversations 中的信息。

主要是和 prompts 模块中的 `ChatMessagePromptTemplate` 和 `ChatPromptTemplate` 搭配使用。

> 此模块只在`langchain_core`模块中有，可以直接使用，不需要在`langchain`等模块中继承。

一、**主要内容**

- `base.py`
  - `BaseMessage`，所有消息的基类——注意，**它并不是抽象类**。
  - `BaseMessageChunk`，所有消息块的基类，大致类似于 `List[BaseMessage]`，这个**也不是抽象类**。

> 实际上 `BaseMessage` 也是 pydantic的`BaseModel`子类。

- `chat.py`, 通用 Message 类
  - `ChatMessage`
  - `ChatMessageChunk`

- `human.py`
  - `HumanMessage`，继承自`BaseMessage`
  - `HumanMessageChunk`，继承自`BaseMessageChunk`

- `system.py`
  - `SystemMessage`,
  - `SystemMessageChunk`

- `ai.py`
  - `AIMessage` 
  - `AIMessageChunk`

- `function.py`
  - `FunctionMessage`
  - `FunctionMessageChunk`

- `tools.py`
  - `ToolMessage`
  - `ToolMessageChunk`

二、**使用说明**

`BaseMessage`/`BaseMessageChunk`两个类已经定义好了重要的属性和方法，其他的Message类大部分都是简单的封装。

（1）常用属性：
- `id`: 消息标识符
- `name`: 消息名称，可选
- `type`: 消息类型，`HumanMessage`/`SystemMessage`等子类会设置这个字段，用于区分不同的消息类型。
- `content`: 消息内容——最重要的部分
- `role`: 消息角色，这个字段只有`ChatMessage`中有，其他子类没有。
- `model_config`:
- `response_metadata`:

（2）常用方法：
- `text`: 获取消息内容(`BaseMessage.content`)，返回一个字符串。 
- `pretty_repr(html: bool = False)`:
- `pretty_print()`:

（3）说明：

需要注意的是，`ChatMessage` 是通用 Message 封装类，有一个 `role` 属性。

而 `SystemMessage`/`HumanMessage`等专用Message封装类没有这个属性，也能正常调用，是因为根据 `SystemMessage`/`HumanMessage` 来判断 role 类型的逻辑放在了具体的 `BastChatModel` 实现类中。

举例来说：

- `ChatOllama._generate()` 方法里，最终会调用 `_convert_messages_to_ollama_messages()` 方法，其中就有 role 的判断逻辑；
- `ChatOpenAI._generate()` 方法里，最终会调用 `_convert_message_to_dict()` 方法，其中有 role 的判断逻辑。


### `prompts`模块

一、**主要内容**
- `base.py`
  - `BasePromptTemplate`, 所有Completion prompt模板的基类，这是个**抽象类**，不能直接使用。

- `message.py`
  - `BaseMessagePromptTemplate`，所有ChatModel的 Message prompt模板的基类，这是个**抽象类**，不能直接使用。

- `string.py`
  - `StringPromptTemplate`，继承自`BasePromptTemplate`，也是个**抽象类**，不能直接使用。

- `prompt.py`
  - `PromptTemplate`, 继承自`StringPromptTemplate`，这个类是**最基础的prompt模板，适用于Completion任务（普通的LLM模型）**。

- `few_shot.py`
  - `FewShotPromptTemplate`
  - `FewShotChatMessagePromptTemplate`

- `few_shot_with_templates.py`
  - `FewShotPromptWithTemplates`

- `pipeline.py`
  - `PipelinePromptTemplate`

- `chat.py`, 定义了ChatModel使用的Prompt模板
  - `BaseStringMessagePromptTemplate`: 继承了`message.py`里的`BaseMessagePromptTemplate`抽象类，它本身也是抽象类。      
    以下是常用实现类：
    - `ChatMessagePromptTemplate`: 专门用于生成符合对话格式的消息（如用户消息、AI 回复、系统提示等）.
      - 主要用于生成**单个**对话消息模板
      - 返回的是 `ChatMessage` 对象
      - **通用**模板类，用于创建包含特定角色（如用户、AI或系统）的消息模板。它允许你指定消息的角色，并通过占位符动态插入变量内容。
    - `HumanMessagePromptTemplate`
      - 生成**单个**对话消息模板，专门为创建用户（人类）消息而设计的一个特化版本的模板类，返回的是 `HumanMessage` 对象
    - `AIMessagePromptTemplate`
    - `SystemMessagePromptTemplate`
  - `MessagesPlaceholder`: 占位符，用于在 ChatPrompt 中插入一个变量，这个变量是一个列表，列表中的每个元素都是一个 `BaseMessage` 对象。
  - `BaseChatPromptTemplate`: 它继承自 `base.py` 的 `BasePromptTemplate`，也是抽象类，只有下面一个实现类
    - `ChatPromptTemplate`: 用于组合 多个ChatMessagePromptTemplate 或者其他类型的提示模板（例如文本提示模板），形成一个完整的对话上下文。
      - 持有一个 `messages` 属性，类型是`List[Union[BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate]]` 

> 注意：
>
> `ChatMessagePromptTemplate`返回的是`ChatMessage`，有 role 属性，type属性是'chat';
>
> `HumanMessagePromptTemplate`返回的是`HumanMessage`，type属性是'human'，**没有 role 属性**。


二、**使用说明**

所有继承`BasePromptTemplate`的类，需要关注如下方法：
- `invoke`/`ainvoke`: 返回值是`PromptValue`及其子类。
- `format_prompt`/`afomat_prompt`: 返回值是`PromptValue`及其子类。
- `format`/`afomat`: 直接返回字符串
- `save`

`BaseMessagePromptTemplate`类虽然是ChatModel的模板基类，但是一般主要使用它的子类`BaseStringMessagePromptTemplate`，需要关注如下方法：
- `format_messages`/`afomat_messages`: 返回值是`List[BaseMessage]`
- `format`/`afomat`: 返回值是`BaseMessage`
- `pretty_repr`/`pretty_print`


### `prompt_values.py`

封装了 Prompt Template 的输出值。

一、**主要内容**
- `PromptValue`: 封装了 prompt 的输出值，这个类是一个**抽象类**
  - 它继承自`Serializable` —— 也是pydantic的`BaseModel`子类，它也是下面所有类的基类。
  - 主要定义了两个方法：
    - `def to_messages(self) -> list[BaseMessage]`
    - `def to_string(self) -> str`

- `StringPromptValue`: 继承自`PromptValue`，用于封装字符串类型的 prompt 输出值。     
  - 有如下属性：
    - `type`: str类型，固定为`StringPromptValue`
    - `text`: str类型，存放具体提示文本
  - `to_message()` 方法返回的是 `List[HumanMessage]`

- `ChatPromptValue`: 继承自`PromptValue`，用于封装ChatMessage类型的 prompt 输出值。    
  - 有如下属性：
    - `messages`: `List[BaseMessage]`类型，存放具体提示消息
  - `to_messages()` 方法直接返回上面的 `messages` 属性

- `ChatPromptValueConcrete`: 继承自`ChatPromptValue`

**使用说明**

`PromptValue` 类及其子类的作用是封装提示词模板的输出，以便适用于不同的任务，使用时主要关注两个方法：
- `to_string() -> str`: 将提示模板转换为一个纯字符串，这是大多数**基础对话LLM**期望的输入格式。
- `to_messages() -> List[BaseMessage]`: 将提示模板转换为一个消息对象列表，这是**聊天模型**期望的输入格式。


### `output` 和 `output_parsers` 模块

`output`模块用于封装LLM输出的内容。
- `chat_generation.py`
  - `ChatGeneration`
  - `ChatGenerationChunk`
- `chat_result.py`
  - `ChatResult`
- `generation.py`
  - `Generation`
  - `GenerationChunk`
- `llm_result.py`
  - `LLMResult`

`output_parsers`模块用于解析LLM输出的内容。
- `base.py`
  - `BaseLLMOutputParser`: 所有 Parser 的抽象基类
  - `BaseGenerationOutputParser`
  - `BaseOutputParser`
- `string.py`
  - `StrOutputParser`
- `json.py`
  - `JsonOutputParser`
- `list.py`
  - `MarkdownListOutputParser`
- `openai_function.py`


---------------------------------------------------
## Memory相关

官方文档[How to migrate to LangGraph memory](https://python.langchain.com/docs/versions/migrating_memory/)建议**转向使用 LangGraph**.

根据上面的官方文档，Langchain 里有关 Memory 的设计思路经历了3个阶段：
1. 基于 `BaseMemory` (`langchain_core.memory.py`) 的早期设计
2. 基于 `RunnableWithMessageHistory` (`langchian_core.runnables.history.py`) 或 
   `BaseChatMessageHistory` (`lanchain_core.chat_history.py`) 的设计，这个设计思路还在沿用，适用于简单的场景
3. 基于 LangGraph 的思路，这个是后续的发展方向

`BaseChatMessageHistory` 是和 `langchain.memory` 模块的 `ChatBaseMemory` 配合使用的，大致流程是 `ChatBaseMemory` 会将历史聊天记录的存储委托给某个 `BaseChatMessageHistory` 实现类来进行。

`RunnableWithMessageHistory` 的使用方式不一样，它是**为了和 LangGraph 配合使用，并且支持 LCEL 表达式**。

LangGraph支持多用户的聊天记录管理，也支持容错恢复功能。


### ~~`memory.py`~~

> **从langchain v0.3.3 版本开始，memory模块被表示为废弃，并在 v1.x 版本被移除了**。  

只提供了一个类：`BaseMemory`，所有memory的基类，提供了一些通用的接口。   

`BaseMemory`继承了`Serializable`，所以也是一个Pydantic的`BaseModel`子类。

`BaseMemory`定义了如下抽象方法：

- `memory_variables`: 返回`list[str]`，表示此memory提供了哪些key给模型使用。
- `load_memory_variables`/`aload_memory_variables`: 返回一个字典
- `save_context`/`asave_context`: 保存上下文的输入和输出信息
- `clear`/`aclear`: 清空上下文信息


### `chat_history.py`

一、**主要内容**
- `BaseChatMessageHistory`: 用于表示聊天历史记录的抽象基类
- `InMemoryChatMessageHistory`: 存放在内存中的聊天历史记录简单实现类

`BaseChatMessageHistory`定义了一个属性`messages: list[BaseMessage]`，还定义了如下抽象方法：
- `add_message`: 用于添加消息
- `add_messages`/`aadd_messages`: 用于批量添加消息
- `add_user_message`/`add_ai_message`: 用于添加用户/AI消息
- `aget_messages`: 异步获取历史消息
- `clear`/`aclear`: 清空历史消息

`InMemoryChatMessageHistory`就是一个简单的基于内存列表的`BaseChatMessageHistory`实现类。

二、**使用说明**

`BaseChatMessageHistory`有两种使用方式：
1. 配合`langchain.memory.chat_memory.py`里的`BaseChatMemory`一起使用的，这种方式已经不太推荐了。
2. 配合下面的 `RunnableWithMessageHistory`一起使用 —— 这种方式比较推荐。


### `runnables.history.py`

此文件里定义了 `RunnableWithMessageHistory` 类，**它和上面的 memory 模块、chat_history 模块的使用方式差异很大**。

`RunnableWithMessageHistory` 主要作用是对一个 `Runnable`对象 和它的 对话历史 进行封装管理。

为了管理对话历史，它要求在每次调用时，都提供一个 `session_id` ，用于确定内部的 `Runnable` 对象的对话历史。

它初始化时，需要提供如下参数：
- `runnable: Runnable`: 需要被包装的 `Runnable` 对象
  - 输入是 `list[BaseMessage]`
  - 返回值是 `str | BaseMessage | MessagesOrDictWithMessages`

- `get_session_history`: 一个Callable对象，它的参数一般是一个`session_id`，然后返回该session的 `BaseChatMessageHistory` 对象
- `input_messages_key`:
- `output_messages_key`:
- `history_messages_key`:
- `history_factory_config`:


---------------------------------------------------
## 数据检索（RAG）相关

LangChain 中将数据检索（RAG）分为以下几个步骤：

1. Loader: 加载器，用于加载Document数据
2. Documentation Transform: 对Document进行转换，也就是 Text-Splitter，生成 Chunks
3. Embedding: 向量嵌入，生成Document/Chunks的Text Embedding向量 
4. VectorStore: 向量数据库，用于存储Document/Chunks的Text Embedding向量 
5. Retriever: 向量检索器，统一封装VectorStore的检索功能

> 注意，langchain-core 里的以下模块，从 v0.3.x 到 v1.x 版本的变化不大。
>
> 相比之下，**LangChain提供的RAG能力比LlamaIndex弱不少**。


### `documents`模块

定义了LangChain里的文档对象的通用表示。

一、**主要内容**

`base.py`
- `class BaseMedia(Serializable)`: 所有 Media 的抽象基类，Media 包括text
- `class Blob(BaseMedia)`: 文档的二进制数据（raw data）
- `class Document(BaseMedia)`: 文档的基类，包含text和metadata —— KEY

`compressor.py`

- `BaseDocumentCompressor`

`transformers.py`

- `class BaseDocumentTransformer(ABC)`，抽象类，作为接口使用，只有如下两个接口方法
  - `def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]`，这是抽象方法
  - `async def atransform_documents()`，异步接口，非抽象方法，内部会使用 `run_in_excutor()` 封装调用上面的方法。


二、**使用说明**

`BaseMedia`继承自 `Serializable`，所以也是一个Pydantic的`BaseModel`子类，里面定义了如下两个属性：
- `id`: 可选str，用于标识文档
- `metadata`: dict，用于存储文档的元数据

`Document`继承自 `BaseMedia`，新增如下两个属性：

- `type`: 固定是 Document
- `page_content`: str，文档的文本内容


### `document_loaders`模块

> 还有一个 `load` 模块，该模块提供了序列化和反序列化相关的工具.

一、**主要内容**

`base.py`
- `BaseLoader`: 所有 Loader 的抽象基类——定义了统一接口
- `BaseBlobParser`: 

`blob_loaders.py`

- `BlobLoader`

`langsmith.py`

- `LangSmithLoader`

二、**使用说明**

`BaseLoader`里定义了如下接口：
- `load`/`aload`: 加载数据，返回 `List[Document]`
- `lazy_load`/`alazy_load`: 迭代加载数据，返回 `Iterator[Document]`
- `load_and_split`: 加载并分割数据，返回 `List[Document]`

这几个方法也是所有Loader的通用方法。


### `embeddings`模块

一、**主要内容**
- `embeddings.py`: 只有一个 `Embeddings` 抽象基类

二、**使用说明**   

`Embeddings`是一个抽象基类，定义了如下方法：

- `embed_query`/`aembed_query`: 用于计算query的embedding向量，返回一个 `List[float]`
- `embed_documents`/`aembed_documents`: 用于**批量计算**query的embedding向量，返回一个 `List[List[float]]`

`Embeddings`类没有定义任何属性，因此实现方式全看子类。


### `vectorstores`模块

一、**主要内容**

`base.py`
- `VectorStore`: 所有 VectorStore 的抽象基类
- `VectorStoreRetriever`: 所有 VectorStoreRetriever 的抽象基类，它继承自 `retrievers.py`里的`BaseRetriever`

`in_memory.py`
- `InMemoryVectorStore`

`utils.py`

二、**使用说明**   

`VectorStore`抽象基类里定义了如下方法：
- `add_texts`/`aadd_texts`
- `add_documents`/`aadd_documents`
- `delete`/`adelete`
- `get_by_ids`/`aget_by_ids`
- `search`/`asearch`
- `similarity_search`/`asimilarity_search`
- `as_retriever`: 返回一个 `VectorStoreRetriever` 对象，这个方法比较实用 —— KEY


`VectorStoreRetriever`抽象基类里定义了如下方法：
- `add_documents`/`aadd_documents`
- `add_documents`/`aadd_documents`


### `retriever.py`

定义了 `BaseRetriever`类，继承自 `RunnableSerializable`，因此也是通过通用方法 `invoke()`/`ainvoke()` 进行调用。


---------------------------------------------------
## Agent相关

### `tools`模块

一、**主要内容**
- `base.py`
  - `BaseTool`: 所有工具类的抽象基类，它继承了 `RunnableSerializable`，所以也是一个Pydantic的`BaseModel`子类。    
     它定义了Langchain里Tool需要实现的接口，只有一个抽象方法`_run()`（异步版本的`_arun()`方法底层调用的也是这个）需要子类实现具体的工具调用逻辑
  - `BaseToolkit`: 所有工具集类的抽象基类，它没有继承 `BaseTool`，不过继承了Pydantic的`BaseModel`。
- `simple.py`
  - `Tool`: 工具类，继承自 `BaseTool`，实现了`_run()`方法。
- `structured.py`
  - `StructuredTool`: 结构化工具类，继承自 `BaseTool`，实现了`_run()`方法 —— 推荐使用这个。
- `convert.py`: 提供了`@tool`装饰器，用于将函数转换为工具类（`StructuredTool`对象或者`Tool`对象）。
- `render.py`
- `reriever.py`

二、**使用说明**

`BaseTool`类里定义了如下属性（对应于function calling所必须的3个要素）：

- `name`: str类型，工具类的名称，用于标识工具类的唯一性，必须唯一。
- `description`: str类型，工具类的描述，用于标识工具类的用途。
- `args_schema`: Pydantic的`BaseModel`子类，用于定义工具类的参数，如果定义了该属性，则该工具类将支持参数校验。

其他属性：
- `return_direct`: bool类型，表示是否直接返回结果，如果为True，则直接返回结果，如果为False，则返回一个字典，字典的key为`output`，值为结果。
- `verbose`: bool类型，表示是否打印日志，如果为True，则打印日志，如果为False，则不打印日志。
- `callback_manager`: CallbackManager类型，用于管理回调函数，如果为None，则使用默认的回调管理器。
- `metadata`: dict类型，表示工具类的元数据，用于标识工具类的用途。
- `tags`:
- `handle_tool_error`: 
- `handle_validation_error`:

定义了如下调用方法：
- `run`/`arun`: 用于执行工具类内部原生函数的调用
- `invoke`/`ainvoke`: 对`run`/`arun`的封装，满足`RunnableSerializable`接口的要求，建议通过这两个方法调用。
- Callable调用，不过后续可能不再支持


### `agents.py`

定义了如下类：

- `AgentAction`: 表示Agent发起的执行请求，是一个数据类，有如下属性：
  - `tool`: 请求执行的工具名称
  - `tool_input`: 请求执行的工具输入
  - `log`: 附加日志信息
- `AgentActionMessageLog`: 继承自 `AgentAction` 类，表示
- `AgentStep`
- `AgentFinish`

> Langchain-core里的agents内容并没有太多，主要在langchain包里。




---------------------------------------------------

# LangChain:v0.3

Module名称为`langchain`，源码内容如下：

```text
## langchain v0.3.27 模块及文件
### 子模块
- api
- adapters
- callbacks
- chains
- chat_models
- docstore
- document_loaders
- document_transformers
- embeddings
- evaluation
- graphs
- indexes
- llms
- load
- memory
- output_parsers
- prompts
- pydantic_v1
- retrievers
- runnables
- schema
- smith
- storage
- tools
- utilities
- utils
- vectorstores

### 根目录文件
- __init__.py
- base_language.py
- cache.py
- env.py
- example_generator.py
- formatting.py
- globals.py
- hub.py
- input.py
- model_laboratory.py
- pytyped
- python.py
- requests.py
- sequapi.py
- sql_database.py
- text_splitter.py
```

所有的模块可以分为如下6大类：


## Chain核心模块-KEY

> `langchain_core`没有chains模块，因为 `langchain_core.runnables` 定义的就是chains相关的核心接口/抽象类。
>
> **`chains` 模块是 langchain 包的核心内容**。

### `chains`模块-

一、**主要内容**

- `base.py`:
  - `Chain`: Chain组件的抽象基类，它继承了 `RunnableSerializable`，所以也是一个Pydantic的`BaseModel`子类。

`chains`模块提供了一系列的Chain组件实现类。

二、**使用说明**

抽象基类`Chain`里定义了如下属性（这些属性可以在初始化时传入）：

- `metadata: Optional[Dict[str, Any]] = None`
- `tags: Optional[List[str]] = None`
- `verbose: bool`: 控制是否输出日志
- `memory: Optional[BaseMemory]`: 存储 Memory 对象
- `callbacks`: 回调函数配置，类型是`Union[list[BaseCallbackHandler], BaseCallbackManager]`，既可以是回调函数列表，也可以是`BaseCallbackManager`对象
- `callback_manager: Optional[BaseCallbackManager]`: 回调管理器，和`callbacks`重复了，所以**被标识为废弃的**，建议使用`callbacks`

抽象基类`Chain`定义了如下抽象方法：

- Callable调用: 执行Chain组件
  - 已被标记为废弃，后续不再支持，代替方法是`invoke`/`ainvoke`。
  - 输入的`inputs`是一个`Union[Dict[str, Any], Any]`，应当包含`Chain.input_keys`里的所有key（Memory使用的key除外），如果只有一个参数，则可以直接传入。
  - 返回值是`Dict[str, Any]`，A dict of named outputs，包含了`Chain.output_keys`属性指定的所有key
- `run`/`arun`: 执行Chain组件
  - 已被标记为废弃，后续不再支持，代替方法是`invoke`/`ainvoke`。
  - 和Callable调用的区别是，它接受的输入不是像`__call__`中那样的`Dict[str, Any]`，而是必须拆开以关键字参数的形式传入，如果只有一个参数，则采用位置参数（第1个）传入.
  - 返回值是`Any`，要看具体的`Chain`和配置的LLM
- `save`: 保存Chain，可以传入一个`file_path`
- `dict`: 以dict的形式返回Chain的表示

此外，`Chain`还定义了如下两个抽象property需要子类实现：

- `input_keys`: `List[str]`类型，指定Chain组件的输入key
- `output_keys`: `List[str]`类型，指定Chain组件的输出key
- `_chain_type`: `str`类型，指定Chain组件的类型，不过这个属性一般是内部使用的



`LLMChain`



### `callbacks`模块



---------------------------------------------------
## Model IO模块

> `langchain`包的`llm`/`chat_models`模块里都只是提供了模型定义、加载初始化的内容，返回的模型都是 `langchain_core.language_model`模块的抽象类及其子类。研究调用过程，应该看`langchain_core.language_model`模块里的源码。

### `llms` 模块

提供LLM类型的具体实现模型，不过这里面的模型都是从`langchain_community.llms`模块导入，源码里没有啥内容。

相比于 ChatModel，LLM类型的模型用的没有那么多。

### `chat_models`模块

和上面的LLM类似，该模块里的模型都是从 `langchain_community.chat_models` 模块导入的。

不过ChatModel模块（`base.py`里）提供了一个 `init_chat_model` 函数，用于初始化ChatModel相关的配置。

`init_chat_model()`函数有4种重载的形式，具体参考源码，其中比较重要的参数如下：
- `model`: LLM模型名称，如`gpt-3.5-turbo`
- `model_provider`: LLM模型提供者名称，如`openai`，一般会对应一个 `langchain-{provider}`包
- `configurable_fields`
- `config_prefix`
- `temperature`
- `max_tokens`
- `timeout`
- `max_retries`
- `base_url`
- `rate_limiter`
- `kwargs`: 模型初始化的其他参数，依具体模型而定

### `prompts`模块

这个模块写的稍微有点敷衍，就是把 `langchain_core.prompts`模块里的对应内容导入过来。


### `output_parsers`模块

这个模块也是把 `langchain_core.output_parsers` 里的内容导入过来。




---------------------------------------------------
## Memory 相关模块

### `memeory`模块

这个模块混合了早期 **基于`BaseMemory`实现** 思路的Memory组件和 **基于`BaseChatMessageHistory`实现** 思路的Memory组件。

一、**主要内容**

提供了**基于`BaseMemory`实现**的（常用）内容：
- `simple.py`:
  - `SimpleMemory`: 用于（初始化）存储一个固定的聊天记录，只能读，不能修改或者清除。
- `readonly.py`:
  - `ReadOnlySharedMemory`: 对一个`BaseMemory`对象进行只读的包装，不能修改，不能清除。
- `chat_memory.py`:
  - `BaseChatMemory`: Chat类型Memory的抽象基类
- `buffer.py`: 提供了对话类的 Memory
  - `ConversationBufferMemory`
  - `ConversationStringBufferMemory`
- `buffer_window.py`:
  - `ConversationBufferWindowMemory`
- `summary.py`: 提供了摘要类的 Memory
  - `ConversationSummaryMemory`
- `summary_buffer.py`:
  - `ConversationSummaryBufferMemory`
- `token_buffer.py`:
  - `ConversationTokenBufferMemory`
- `combined.py`:


二、**使用说明**

`BaseChatMemory`抽象类继承了`BaseMemory`类，并在此基础上定义了如下属性：
- `chat_memory: BaseChatMessageHistory`: 这个就是配合下面的`BaseChatMessageHistory`使用的，它的默认实现就是`InMemoryMessageHistory`。
- `output_key: Optional[str] = None`:
- `input_key: Optional[str] = None`:
- `return_messages: bool = False`:

**`BaseChatMemory` 底层会将历史消息的读写操作委托给 `BaseChatMessageHistory` 实现类**，具体过程是：`save_context`方法里调用`chat_memory: BaseChatMessageHistory`属性对象的`add_messages`方法。

**基于`BaseChatMessageHistory`实现** 的组件在`langchain.memory.chat_message_history`模块里。

> 此模块其实只是一个简单导入的封装，实际是从 `langchain_community.chat_message_histories` 模块导入对应的类。

常用的ChatMessageHistory实现类如下：
- `ChatMessageHistory`: 这个其实就是 `langchain_core.chat_history.py` 里 `InMemoryChatMessageHistory` 的别名
- `FileChatMessageHistory`: 基于本地文件实现
- `RedisChatMessageHistory`
- `SQLChatMessageHistory`
- `ElasticsearchChatMessageHistory`



------


## 数据检索（RAG）相关模块

> 以下模块，从 v0.3.x 升级到 v1.x 版本，除了 `embeddings` 模块依旧保留外，其他模块都移动到 `langchain_community`包了。
>
> 实际上，即使是在 v0.3.x 模块，以下大部分模块也都是从 `langchain_community`包里导入的功能。
>
> 由于 `langchain_community` 包并没有升级到 v1.x，所以可以认为 RAG 相关的模块变化不大。

### `document_loaders`模块

这个模块主要从两个地方导入内容：

- `langchain_core.document_loaders`里导入 `BaseLoader` 和 `BaseBlobParser`
- `langchain_community.document_loaders`里导入各种类型的 Loader 实现类

> langchain官方建议后续直接从 `langchain_community`包里导入。

个人感觉常用的有如下Loader实现类：

- TextLoader
- CSVLoader
- JSONLoader
- WebBaseLoader
- DataFrameLoader
- HuggingFaceDatasetLoader
- PyMuPDFLoader
- PyPDFDirectoryLoader
- PyPDFium2Loader
- PyPDFLoader
- BiliBiliLoader, 居然还有B站

### `document_transformers`模块

主要从 `langchain_community.document_transformers` 里导入各种文档转换器。


### `embeddings`模块

这个模块主要从两个地方导入内容：

- `langchain_core.embeddings`里导入 `Embeddings` 抽象基类
- `langchain_community.embeddings`里导入各种类型的 Embeddings 实现类

> langchain官方建议后续直接从 `langchain_community`包里导入。

每一个Embeddings实现类都继承自 `Embeddings`抽象基类，并且继承了 Pydantic的 `BaseModel`子类。

不过每个Embeddings实现类初始化时配置模型的参数好像都不太一样，具体需要参考对应的实现类的文档或源码。

常用的Embeddings实现类：

- OpenAIEmbeddings
- HuggingFaceEmbeddings
- OllamaEmbeddings


### `vectorstores`模块

和上面类似，这个模块主要从两个地方导入内容：

- `langchain_core.vectorstores`里导入 `VectorStore` 抽象基类
- `langchain_community.vectorstores`里导入各种类型的 VectorStore 实现类

> langchain官方建议后续直接从 `langchain_community`包里导入。

### `retriever` 模块

和上面类似，这个模块主要从两个地方导入内容：

- `langchain_core.retriever`里导入 `BaseRetriever` 抽象基类
- `langchain_community.retriever`里导入各种类型的 BaseRetriever 实现类

> langchain官方建议后续直接从 `langchain_community`包里导入。




---------------------------------------------------
## Agent相关模块

> LangChain的 Agent 相关模块在 0.3 版本之后有较大改动：
>
> - `langchain.agents`模块里内容是之前构建Agent的方式，已被标记为废弃的，在 1.0.0 版本之前都会被保留，参见官方文档 [How to migrate from legacy LangChain agents to LangGraph](https://python.langchain.com/docs/how_to/migrate_agent/).
> - 后续LangChain官方推荐转向使用 LangGraph 构建 Agent 应用，因此这里就**不再详细介绍 agents 模块相关内容了**。

### `tools`模块

> v1.x 版本的 tools 模块改动很大，几乎相当于重构了。

和上面类似，这个模块主要从两个地方导入内容：
- `langchain_core.tool`里导入抽象基类
- `langchain_community.tool`里导入各种实现类

> langchain官方建议后续直接从 `langchain_community.tool`包里导入。

各个感觉比较常用的一些Tool如下：
- ListDirectoryTool
- ReadFileTool
- WriteFileTool
- CopyFileTool
- MoveFileTool
- DeleteFileTool
- FileSearchTool
- ExtractTextTool
- HumanInputRun
- ShellTool
- GoogleSearchRun
- GoogleSearchResults
- JsonGetValueTool
- JsonListKeysTool
- BaseRequestsTool
- BaseSQLDatabaseTool
- BaseSparkSQLTool
- ListSQLDatabaseTool
- ListSparkSQLTool
- RequestsDeleteTool
- RequestsGetTool
- RequestsPatchTool
- RequestsPostTool
- RequestsPutTool


### `agents`模块

旧版本构建Agent的方式，已经**被标记为废弃**，后续不再支持，建议使用 LangGraph 构建 Agent 应用。



---------------------------------------------------
# LangChain-Community

LangChain-Community是一个第三方社区扩展包，提供了一些常用的功能。

`langchain_community` 包并没有随着 LangChain-Core 和 LangChain 一起升级到 v1.x 版本，而是升级到 v0.4.x 版本。

查看两个版本的源码内容也可以发现，两者变化并不大。



---------------------------------------------------
# LangGraph

> LangGraph 从 **v0.6.x** 版本升级到 **v1.x** 版本，核心组件的变化不大。

首先要明确的是，LangGraph并不依赖LangChain-Core或者LangChain，参考官方[FAQ -> Do I need to use LangChain to use LangGraph? What’s the difference?](https://langchain-ai.github.io/langgraph/concepts/faq/#do-i-need-to-use-langchain-to-use-langgraph-whats-the-difference)。

LangGraph更像是一个高度抽象的基于图的Agent调度框架，LangGraph的核心抽象 Graph 有如下3个概念（ [LangGraph Glossary](https://langchain-ai.github.io/langgraph/concepts/low_level/)）：

- `State`: 图的状态，这是 Graph 的核心，本质上就是一个dict，不过通常用 `TypeDict` 类或者 Pydantic的`BaseModel`类表示。
- `Nodes`: Graph 的计算节点，本质上就是一个Python函数（更广泛一点就是一个Callable对象），可以封装各种逻辑，比如langchain里LLM/ChatModel/Chain的调用。
- `Edges`: Graph 的边，用来连接节点，本质上也是一个Python函数，用于根据当前的`State`判断并返回下一个要执行的`Node`的名称。

> 简单来说，Node是实际执行计算的抽象，Edge是控制逻辑的抽象。


个人理解，LangGraph是基于图的workflow，不同于常用的图计算的场景，它主要**使用图来描述执行的 Workflow**，关注点是`State`对象。

**`State`被视为整个 Graph Workflow 在某一时刻的状态快照**，原因有如下几点：

- 它作为整个 Graph Workflow 的初始输入
- workflow的每一步`Node`，都会接受上一个节点的 `State` 作为输入
- 每个`Node`执行完毕后，都会更新（或者不更新）`State`里的状态
- 最终输出也是`State`，不过此时其中的状态（也就是字段）已经是经过计算后的最终结果了

> `State`里的属性字段完全是用户自己定义的，只要用户自己在Graph的Node/Edge里自己约定就行，所以非常灵活，高度可定制。

在上面的 Graph Workflow 框架的基础上，LangGraph还提供了如下的功能：
- Checkpoint机制: 对应的就是Memory，每一步（Node）执行完都会保存当前Node的`State`，以方便可以作为恢复的快照，另一方面也是作为历史消息
- Interrupt/Command机制：也就是打断/恢复功能，可以方便的添加人工介入的步骤，校验/纠正Agent的执行过程，或者获取人工反馈以进行下一步执行
- TimeTravel机制：可以方便的回溯到之前的某个节点，重新执行，或者重新执行整个Graph，这个依赖的就是Checkpoint机制


---------------------------------------------------
## 两类API

LangGraph提供了两种API:
- [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api): 基于底层的图结构来定义Agent
- [Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api): 对已有的函数进行封装，定义Agent

两类API的选择可以参考官方文档 [Choosing between Graph and Functional APIs](https://docs.langchain.com/oss/python/langgraph/choosing-apis).

简单总结如下：
- Graph API 适合复杂的场景，需要高度自定义
- Functional API 适合简单的场景，特别是将LangGraph应用到已有的函数上，并做最小的改变。

> 个人感觉，推荐使用 Graph API, Functional API 的局限性比较大。

---------------------------------------------------
## `constants.py`

LangGraph的常量字符串定义，这些字符串使用了`sys.intern()`函数驻留内存，避免重复创建。

常用常量如下：
- `START`
- `END`


---------------------------------------------------
## `config.py`


---------------------------------------------------
## `graph`模块 - KEY

### ~~`graph.py`~~

> 这个文件只有 v0.4.10 版本之前有，v0.5.0版本开始就删除了此文件，也没有了`Graph`和`CompiledGraph`类。

无状态图的表示，定义了如下3个类：

- `NodeSpec`: 继承自 `NamedTuple`，用来表示一个节点，有如下属性：
  - `runnable`: 此Node对应的 `Runnable`对象
  - `metadata`: 
  - `ends`:

- `Graph`: 无状态图表示

- `CompiledGraph`: 继承自`pregel`模块的`Pregel`类，`Graph.compile`方法返回的就是此类的对象。

**使用说明**

`Graph`类用于表示无状态图，它使用如下属性来存储图的信息：

- `nodes: dict[str, NodeSpec] = {}`
- `edges = set[tuple[str, str]]()`

`Graph`类提供了如下方法：

- `add_node`
- `add_sequence`
- `add_edge`
- `add_conditional_edges`
- `set_entry_point`/`set_finish_point`: 设置起始/结束节点，快捷方法，内部调用了`add_edge`方法
- `compile`: 编译图，返回一个`CompiledGraph`对象

注意，**`Graph`类的初始化方法不接受任何参数，所以说它是无状态的**。

`CompiledGraph`是`Graph`编译后的对象，它继承了`pregel`模块的`Pregel`类，提供了如下方法（`Pregel`定义的）：
- `invoke`/`ainvoke`
- `stream`/`astream`
- `get_state`/`aget_state`
- `update_state`/`aupdate_state`
- `get_state_history`/`aget_state_history`

------
### `state.py` - KEY

有状态图的表示，定义了如下2个类：

- `StateGraph`:
- `CompiledStateGraph`

> 在v0.4.10版本以前，`StateGraph`继承自 `Graph`，`CompiledStateGraph`继承自`CompiledGraph`，但是从v0.5.10版本开始，这两个类就不再继承父类了。

**使用说明**

`StateGraph` 有如下 4 个需要重点关注的属性，也是初始化时需要传入的参数：

- `state_schema: StateT`: **最重要的属性**，定义了整个Graph的状态，可以是 `TypeDict` / `dataclass` / `pydantic.BaseModel`
- `context_schema: ContextT`: 定义了LangGraph的 `Runtime[ContextT]` 里的泛型参数 `ContextT`，这个类就是自定义的运行时上下文类。
- `input_schema: InputT`: 定义了 LangGraph 的输入结构，可以是 `TypeDict` / `dataclass` / `pydantic.BaseModel`
- `output_schema: OutputT`: 定义了 LangGraph 的输出结构，可以是 `TypeDict` / `dataclass` / `pydantic.BaseModel`

**对于 `input_schema` 和 `output_schema` 参数，如果不设置，则默认使用 `state_schema`**。

上面这个四个参数，其实对应的就是 `StateGraph[StateT, ContextT, InputT, OutputT]` 的 4 个泛型类。

`CompiledStateGraph[StateT, ContextT, InputT, OutputT]` 的 4 个泛型参数和 `StateGraph` 的 4 个泛型参数一样。

------
### `message.py`

定义了如下2个类：

- `MessagesState`: 继承自 `TypedDict`。    
  定义了一个 `message` 属性，类型是`list[AnyMessage]`，并用`Annotated`注解设置了一个reducer函数`add_messages`。

- `MessageGraph`: 继承自 `StateGraph`

还定义了一个常用的reducer函数：`add_messages`。


---------------------------------------------------
## `types.py` - KEY

此源文件里定义了LangGraph 里重要的数据类型。

常用的有如下数据类型：

### `StateSnapshot`

checkpoint里快照的数据结构封装。

### `Interrupt`

触发中断时的数据结构封装，是一个`dataclass`类。
- 封装了如下属性：
  - `id`: 标识此次中断的标识符，一般不需要手动生成
  - `value`: 此次中断的附加信息，用户传入，可以是任何对象
- 一般推荐通过类方法 `from_ns(value: Any, ns: str)` 来创建，它会自动生成一个唯一的id。 

### `Send`

配合 `add_conditional_edges()` 方法使用的数据结构封装，用于动态生成条件边。

- 一般由 `add_conditional_edges()` 里的 `path` 参数传入的函数返回 一个或者多个 `Send` 对象
- 这个类很简单，就是一个单纯的数据封装，只有两个属性：
  - `node`: 指定要执行的节点名称
  - `args`: 传递给目标节点的 state / message
- `Send`类**没有定义任何方法**，所有的处理逻辑都交给在Graph实现。

### `Command`

用于状态更新和动态控制流（相当于conditional_edges）的数据结构封装。 

> 官方文档 [Graph-API -> Command](https://docs.langchain.com/oss/python/langgraph/graph-api#command)
> 
> 使用示例可以参考官方文档 [Graph-API -> Use the Graph API -> Combine control flow and state updates with Command](https://docs.langchain.com/oss/python/langgraph/use-graph-api#combine-control-flow-and-state-updates-with-command)

`Command` 是一个`dataclass`类，定义了如下属性：
- `graph: str`，指定要发往的 graph：
  - `None`，表示当前 graph
  - `str`，一般是`Command.PARENT`，用于Multi-Agent场景，表示父级 graph
- `update: Any`，表示要更新的状态
- `resume: dict[str, Any] | Any`，中断后恢复执行的信息，这个参数的值会被作为 `interrupt()` 方法的返回值
- `goto: Send | Sequence[Send | N]`，要跳转的节点
  - `str` / `List[str]`，指定（多个）跳转Node的name
  - `Send` / `List[Send]`，使用 `Send` 对象来封装要跳转的节点和参数。

`Command`有如下几个使用场景：

- 作为 Node 节点的返回值 —— 最常见
- 作为中断恢复执行的传入参数 —— 也常见

`Command` 和 conditional_edges 的使用区别在于：

- 如果要同时更新state，并根据state执行动态控制流，使用 `Command`
- 如果只是设置节点之间的控制流，使用 conditional_edges


---------------------------------------------------
## `checkpoint`模块 - KEY

此模块对应的是LangGraph里的**短期记忆机制**，只维护每次会话内的历史消息记录。

### `base`子模块

定义了`CheckpointTuple`，继承于`NamedTuple`，用来表示一个状态快照，有如下属性：
- `config: RunnableConfig`
- `checkpoint: Checkpoint`
- `metadata: CheckpointMetadata`
- `parent_config: Optional[RunnableConfig] = None`
- `pending_writes: Optional[List[PendingWrite]] = None`


定义了`BaseCheckpointSaver`基类，用来保存和加载状态快照。


### `memory`子模块

实现了一个`InMemorySaver`，基于内存来保存checkpoint。

### `serde`子模块

定义序列化/反序列化相关内容。



---------------------------------------------------
## `store`模块 - KEY

此模块对应于 LangGraph 的**长期记忆机制**，用于保存和加载长期记忆。

### `base`子模块

`BaseStore`: 所有Store类的抽象基类，定义了如下方法：

- `put`/`aput`
- `get`/`aget`
- `list_namespaces`/`alist_namespaces`
- `search`/`asearch`
- `batch`/`abatch`
- `delete`/`adelete`

### `memory`子模块

定义了一个`InMemoryStore`，基于内存来保存长期记忆。


---------------------------------------------------
## `prebuilt`模块 - KEY

这个模块提供了一些用于构建 Tool 的预制组件，在 v0.6.x 版本还提供了Agent 的预制组件，但是从 v1.0.0 开始，Agent 的预制组件不推荐使用了。

因此这个模块主要是`tool_node.py`里提供的`ToolNode`相关定义。

### `ToolNode`类

封装所有Tools的节点类，它的初始化签名如下：

```python
class ToolNode(RunnableCallable):
    def __init__(
        self,
        tools: Sequence[BaseTool | Callable],
        *,
        name: str = "tools",
        tags: list[str] | None = None,
        handle_tool_errors: bool
        | str
        | Callable[..., str]
        | type[Exception]
        | tuple[type[Exception], ...] = _default_handle_tool_errors,
        messages_key: str = "messages",
        wrap_tool_call: ToolCallWrapper | None = None,
        awrap_tool_call: AsyncToolCallWrapper | None = None,
    ) -> None:
        ...
```

其中比较重要的几个参数如下：

- `tools`，封装的Tools列表
- `messages_key`，指定状态中的消息的key
- `wrap_tool_call`/`awrap_tool_call`，tool调用前后的封装函数
- `handle_tool_errors`，如何处理工具调用异常，可选值如下：
  - `False`：不处理错误，允许向上抛出异常
  - `True`：捕获所有错误，但不抛出，正常返回`ToolMessage`，但是其中包含错误信息
  - `str`：捕获所有错误，但不抛出，正常返回`ToolMessage`，其中的错误信息使用这里自定义的字符串
  - `type[Exception]`/`tuple[type[Exception], ...]`，指定要捕获的异常类，可以有多个，为这些异常返回默认信息，其他异常继续抛出
  - `Callable[..., str]`：自定义异常处理函数

`ToolNode`内部的`_func`/`_afunc`方法，会**并行处理**返回的消息列表中的多个ToolCall。



### `tools_condition`函数

封装 tool 的条件边，源码里内部逻辑比较简单，就是判断 state 里有没有 messages，并且messages 最后一条是不是 AIMessage，是就调用 Tool，否则转向 END。

### ~~`create_react_agent`~~

由`chat_agent_executor.py`文件定义。

> v1.0版本开始，此函数被标记为废弃的了。

用于快速创建一个React Agent。


此外，还定义了一些注解类型，用于在 tool 函数中访问图的状态和存储。
- `InjectedState`
- `InjectedStore`

---------------------------------------------------
## `runtime.py`

定义了 `Runtime` 泛型类，也是一个 `dataclass` 对象。

> 这个 `Runtime` 类的介绍反倒在 LangChain v1.x 的文档里：[Runtime](https://docs.langchain.com/oss/python/langchain/runtime).

`Runtime[ContextT]` 泛型类定义了如下属性：
- `context`: 这个对象就是泛型类型对象 `ContextT`，它必须是一个 `TypedDict`/`dataclass`/pydantic `BaseModel` 对象。
- `store`: LangGraph的 `BaseStore`
- `stream_writer`: `StreamWriter` 对象，用于流式输出的`custom`模式使用。
- `previous`

这个 `Runtime` 类对象可以在 Tool 函数中获取，用于访问运行时上下文环境。

不过似乎一般使用LangChain里提供的 `langchian.tools.tool_node.py` 里的 `ToolRuntime` 对象比较多。

---------------------------------------------------
## `utils`模块

定义了一些LangGraph里的工具函数。

### `runnable.py`

基于 `langchain_core.runnables.base` 里的 `Runnable`设计模式，定义了LangGraph里的 `Runnable`类。

主要有如下两个类：

- `RunnableCallable`
  - 继承自 `langchain_core.runnables.base` 里的 `Runnable` 抽象类


- `RunnableSequence`


---------------------------------------------------
## `pregel`模块

此模块是LangGraph 的 Runtime 实现，它基于Google的Pregel算法，该算法专门用于大规模的并行图计算。

上面的 `CompiledGraph` 类就继承了此模块提供的 `Pregel` 类。

这个模块应该是 LangGraph 的核心实现，研究起来难度比较高。


---------------------------------------------------
## `channels`模块



---------------------------------------------------
## `managed`模块



------

# DeepAgents

`deepagents`包是LangChain官方配合LangChain-V1.0提供的用于编写复杂Agent的包，它底层基于`langchain`+`langgraph`包进行的构建。

简单看了下`deepagents`的源码，以**v0.5.2版本**为例，源码内容并不多，不像langchain/langgraph那样复杂，主要内容如下：

```shell
#v-0.5.2 版本
deepagents/
├── graph.py    # 这是最核心的文件， creage_deep_agent() 函数就在此
 # 定义了虚拟文件系统，抽象接口协议在 protocol.py 中
├── backends/   
    ├── __init__.py
    ├── composite.py
    ├── filesystem.py
    ├── langsmith.py
    ├── local_shell.py
    ├── protocol.py   # 定义了 backend 类协议
    ├── sandbox.py
    ├── state.py
    ├── store.py
    └── utils.py
 # 提供了一些中间件实现，需要特别关注的是skills.py、subagents.py、async_agents.py
├── middleware/
    ├── __init__.py
    ├── _tool_exclusion.py
    ├── utils.py
    ├── async_subagents.py
    ├── filesystem.py
    ├── memory.py
    ├── patch_tool_calls.py
    ├── permissions.py
    ├── skills.py
    ├── subagents.py
    └── summarization.py
├── profiles/
    ├── __init__.py
    ├── _harness_profiles.py
    ├── _openai.py
    ├── _openrouter.py
├── __init__.py
├── _models.py  # 提供了一些模型解析工具函数
└── _version.py
```



## `graph.py`

这是`deepagents`的核心文件，主要定义了`create_deep_agent()`函数。

### 源码

源码里`create_deep_agent()`的大致逻辑（基于**v0.5.2**版本）如下：

```python
def create_deep_agent(
    # 指定模型
    model: str | BaseChatModel | None = None,
    # 配置可调用的工具
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    # 指定整个 Agent 的系统提示词
    system_prompt: str | SystemMessage | None = None,
    # 配置中间件
    middleware: Sequence[AgentMiddleware] = (),
    # 配置 subagents
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    permissions: list[FilesystemPermission] | None = None,
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    # Agent 名称
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]:
    # ---------------------------------------------------------------------------------
    # 初始化模型
    _model_spec: str | None = model if isinstance(model, str) else None
    model = get_default_model() if model is None else resolve_model(model)
    
    # ---------------------------------------------------------------------------------
    # 这个profile作用有待研究
    _profile = _harness_profile_for_model(model, _model_spec)

    # ---------------------------------------------------------------------------------
    # 对传入的 tools 的description 根据上面的profile进行覆盖改写，暂不清楚此操作是为什么
    # Copy of `tools` with any provider-specific description rewrites.
    # (Tool exclusion is handled by _ToolExclusionMiddleware which filters
    # all tools (user-supplied and middleware-injected) in one place.)
    _tools = _apply_tool_description_overrides(
        tools,
        _profile.tool_description_overrides,
    )
    
    # ---------------------------------------------------------------------------------
    backend = backend if backend is not None else StateBackend()

    # ---------------------------------------------------------------------------------
    # 这是deepagents内置的一个 General-purpose subagent，下面是配置该subagent的spec
    # Build general-purpose subagent with default middleware stack
    gp_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        # 要注意这个中间件，它生成的 TO-list 就是执行的 plan ----------------- KEY
        TodoListMiddleware(),
        FilesystemMiddleware(
            backend=backend,
            custom_tool_descriptions=_profile.tool_description_overrides,
        ),
        create_summarization_middleware(model, backend),
        PatchToolCallsMiddleware(),
    ]
    if skills is not None:
        gp_middleware.append(SkillsMiddleware(backend=backend, sources=skills))

    # Add provider-specific middleware, if any
    gp_middleware.extend(_resolve_extra_middleware(_profile))

    # Strip excluded tools after all tool-injecting middleware has run
    if _profile.excluded_tools:
        gp_middleware.append(_ToolExclusionMiddleware(excluded=_profile.excluded_tools))
    # Prompt caching is unconditional: "ignore" silently skips non-Anthropic models
    gp_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))

    # Permissions must be last so they see all tools from prior middleware
    if permissions:
        gp_middleware.append(_PermissionMiddleware(rules=permissions, backend=backend))

    general_purpose_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": _tools or [],
        "middleware": gp_middleware,
    }
    if interrupt_on is not None:
        general_purpose_spec["interrupt_on"] = interrupt_on

    # ---------------------------------------------------------------------------------
    # 配置用户提供的 subagents 的 spec
    # Set up subagent middleware
    inline_subagents: list[SubAgent | CompiledSubAgent] = []
    async_subagents: list[AsyncSubAgent] = []
    # 遍历用户提供的 subagents
    for spec in subagents or []:
        if "graph_id" in spec:
            # Then spec is an AsyncSubAgent
            async_subagents.append(cast("AsyncSubAgent", spec))
            continue
        if "runnable" in spec:
            # CompiledSubAgent - use as-is
            inline_subagents.append(spec)
        else:
            # 此时用户提供的subagent是以dict形式给出的配置，对这些配置进行处理
            # SubAgent - fill in defaults and prepend base middleware
            raw_subagent_model = spec.get("model", model)
            subagent_model = resolve_model(raw_subagent_model)

            _subagent_spec = raw_subagent_model if isinstance(raw_subagent_model, str) else None
            _subagent_profile = _harness_profile_for_model(subagent_model, _subagent_spec)

            # Resolve permissions: subagent's own rules take priority, else inherit parent's
            subagent_permissions = spec.get("permissions", permissions)

            # Build middleware: base stack + skills (if specified) + user's middleware
            subagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
                # 用户subagent里，也使用了这些中间件
                TodoListMiddleware(),
                FilesystemMiddleware(
                    backend=backend,
                    custom_tool_descriptions=_subagent_profile.tool_description_overrides,
                ),
                create_summarization_middleware(subagent_model, backend),
                PatchToolCallsMiddleware(),
            ]
            subagent_skills = spec.get("skills")
            if subagent_skills:
                subagent_middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
            subagent_middleware.extend(spec.get("middleware", []))

            # Provider-specific middleware for this subagent's model
            subagent_middleware.extend(_resolve_extra_middleware(_subagent_profile))
            if _subagent_profile.excluded_tools:
                subagent_middleware.append(_ToolExclusionMiddleware(excluded=_subagent_profile.excluded_tools))

            # Prompt caching
            subagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
            if subagent_permissions:
                subagent_middleware.append(_PermissionMiddleware(rules=subagent_permissions, backend=backend))

            subagent_interrupt_on = spec.get("interrupt_on", interrupt_on)

            # Inherit parent tools unless the subagent declares its own.
            # Descriptions are rewritten; exclusion is handled by middleware.
            raw_subagent_tools = spec.get("tools") if "tools" in spec else tools
            subagent_tools = _apply_tool_description_overrides(
                raw_subagent_tools,
                _subagent_profile.tool_description_overrides,
            )

            processed_spec: SubAgent = {  # ty: ignore[missing-typed-dict-key]
                **spec,
                "model": subagent_model,
                "tools": subagent_tools or [],
                "middleware": subagent_middleware,
            }
            if subagent_interrupt_on is not None:
                processed_spec["interrupt_on"] = subagent_interrupt_on
            inline_subagents.append(processed_spec)

    # ---------------------------------------------------------------------------------
    # 把内置的 General-Purpose Agent 放到第1位
    # If an agent with general purpose name already exists in subagents, then don't add it
    # This is how you overwrite/configure general purpose subagent
    if not any(spec["name"] == GENERAL_PURPOSE_SUBAGENT["name"] for spec in inline_subagents):
        # Add a general purpose subagent if it doesn't exist yet
        inline_subagents.insert(0, general_purpose_spec)

    # ---------------------------------------------------------------------------------
    # 开始构建 主Agent 的 spec
    # Build main agent middleware stack
    deepagent_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        # 主Agent也使用了这个 Todo 中间件，用于生成 plan
        TodoListMiddleware(),
    ]
    if skills is not None:
        deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    deepagent_middleware.extend(
        [
            FilesystemMiddleware(
                backend=backend,
                custom_tool_descriptions=_profile.tool_description_overrides,
            ),
            # ---------------------------------------------------------------------------
            # 注意，所有的subagent，都是借助这个中间件来插入的 ----------------------------- KEY
            SubAgentMiddleware(
                backend=backend,
                subagents=inline_subagents,
                # Overrides the task tool description. Value should include
                # {available_agents} — a format placeholder replaced with the
                # subagent name/description list. Without it the model can't
                # see which subagents exist. None (default) uses the built-in
                # template. Stale keys silently no-op if the tool is renamed.
                task_description=_profile.tool_description_overrides.get("task"),
            ),
            # ---------------------------------------------------------------------------
            create_summarization_middleware(model, backend),
            PatchToolCallsMiddleware(),
        ]
    )

    if async_subagents:
        # Async here means that we run these subagents in a non-blocking manner.
        # Currently this supports agents deployed via LangSmith deployments.
        deepagent_middleware.append(AsyncSubAgentMiddleware(async_subagents=async_subagents))

    if middleware:
        deepagent_middleware.extend(middleware)
    # Provider-specific middleware goes between user middleware and memory so
    # that memory updates (which change the system prompt) don't invalidate the
    # Anthropic prompt cache prefix.
    deepagent_middleware.extend(_resolve_extra_middleware(_profile))
    if _profile.excluded_tools:
        deepagent_middleware.append(_ToolExclusionMiddleware(excluded=_profile.excluded_tools))
    # Unconditional prompt caching (see general-purpose subagent comment).
    deepagent_middleware.append(AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"))
    if memory is not None:
        deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
    # _PermissionMiddleware must be last so it sees all tools from prior middleware
    if permissions:
        deepagent_middleware.append(_PermissionMiddleware(rules=permissions, backend=backend))

    # ---------------------------------------------------------------------------------
    # 配置基础提示词
    # Assemble base prompt: use _profile.base_system_prompt if set, else
    # BASE_AGENT_PROMPT, then append profile suffix if present.
    # Finally prepend user system_prompt (handled below).
    base_prompt = _profile.base_system_prompt if _profile.base_system_prompt is not None else BASE_AGENT_PROMPT
    if _profile.system_prompt_suffix is not None:
        base_prompt = base_prompt + "\n\n" + _profile.system_prompt_suffix
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = base_prompt
    elif isinstance(system_prompt, SystemMessage):
        final_system_prompt = SystemMessage(content_blocks=[*system_prompt.content_blocks, {"type": "text", "text": f"\n\n{base_prompt}"}])
    else:
        # String: simple concatenation
        final_system_prompt = system_prompt + "\n\n" + base_prompt

    # ---------------------------------------------------------------------------------
    # 直接调用 langchain.agents 模块提供的 create_agent() 函数
    return create_agent(
        model,
        system_prompt=final_system_prompt,
        # tools 还是用户传入的 tools，不过description可能被改写了
        tools=_tools,
        # Agent执行plan生成 + subagent 执行，都是以中间件的形式工作的 -------------------- KEY
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config(
        {
            "recursion_limit": 9_999,  # 限制一下Agent的最大迭代次数
            "metadata": {
                "ls_integration": "deepagents",
                "versions": {"deepagents": __version__},
                "lc_agent_name": name,
            },
        }
    )
```

### 重点说明

所谓的DeepAgent，其中的执行计划生成 + SubAgent，都是以中间件形式实现的。



------

## `profiles`模块



------

## `middleware`模块



------

## `backends`模块





---------------------------------------------------
# LangChain-MCP-Adapters

`langchain-mcp-adapters` 专门为 LangChain 提供 MCP 适配的package，包名为 `langchain_mcp_adapters`。

此模块提供了将 MCP Tools 包装为 LangChain Tools 的功能，同时它**依赖于MCP官方的Python-SDK `mcp` **。

> 以下是基于 **v0.1.14** 版本梳理的模块内容。

`__init__.py` 里没有任何内容，所以无法从顶级模块中导入任何对象。

------
## `client.py` - KEY

定义了一个 `MultiServerMCPClient`，这个类是 MCP-Adapter 大部分情况下的使用入口。

`MultiServerMCPClient`类 有如下 3个 初始化参数：

- `connections: dict[str, Connection]`, MCP服务器连接信息 —— 最重要的配置
- `callbacks`:
- `tool_interceptors`:

`MultiServerMCPClient`类主要提供了如下 3 个方法：

- `get_tools(server_name: str) -> list[Tool]`: 获取指定服务器的 MCP Tools，它会调用下面的 `load_mcp_tools()` 函数。
- `get_resources(server_name: str, uris: str | list[str]) -> list[Blob]`: 获取指定服务器的 MCP Resources，它会调用下面的 `load_mcp_resources()` 函数。
- `get_prompts(server_name: str, prompt_name: str,) -> list[HumanMessage | AIMessage]`: 获取指定服务器的 MCP Prompts，它会调用下面的 `load_mcp_prompts()` 函数。

注意，上面3个方法都是**异步方法**！！！

------
## `sessions.py`

定义了MCP的各类 Connection 的传输协议结构（TypedDict）：
- `StdioConnection`
- `SSEConnection`
- `StreamableHttpConnection`
- `WebsocketConnection`

最重要的是实现了一个 `create_session()` 方法，用于创建一个 MCP Session。

------
## `tools.py`

提供了将 MCP Tools 转换为 LangChain Tools、实现 MCP Tools 调用等功能。

定义了一个 **`load_mcp_tools()`** 函数，用于从 **原生MCP Session** 里获取 MCP Tools 并转换为LangChain-Tools。

------
## `resources.py`

定义了一个 `load_mcp_resources()` 函数，用于从 **原生MCP Session** 里获取 MCP Resources。

------
## `prompts.py`

定义了一个 `load_mcp_prompts()` 函数，用于从 **原生MCP Session** 里获取 MCP Prompts。

------
## `callbacks.py`


------
## `interceptors.py`


------
## 使用说明

有两种使用方式：

（一） 直接使用 `client.py` 里的 `MultiServerMCPClient` 类连接多个MCP服务器：

1. 通过 `get_tools()` 、`get_resources()` 、`get_prompts()` 方法获取 MCP Tools、MCP Resources、MCP Prompts
2. 将这些 MCP Tools、MCP Resources、MCP Prompts 提供给 LangChain/LangGraph 即可。

（二）