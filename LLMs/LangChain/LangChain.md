[TOC]

这里对 LangChain 的各个package进行简单总结。

# Langchain-Core

package名称为`langchain_core`，需要关注的有如下内容。  

大部分模块的说明可以在该模块的 `__init__.py` 文件中找到。

---------------------------------------------------
## Chain基础

这部分的内容是LangChain里的基础，主要用于 Chain 的构建，并支持 LangChain Expression Language (LCEL) 语法。


### `runnables`模块 - KEY

这个模块是langchain_core模块的核心模块，基于 Runnable设计模式 和 LangChain Expression Language (LCEL) 定义了一系列的接口规范。
也是实现 Chain 的核心模块。 

这里重点介绍如下文件里定义的一些常用抽象基类。

#### `base.py`

**`Runnable`**

它是LangChain里大部分对象执行的基本单元对象，是LangChain里的核心抽象基类，
详细介绍可以参考官方文档[Conceptual Guide -> Runnable interface](https://python.langchain.com/docs/concepts/runnables/).

它重载了运算符`|`（重写了`__or__`/`__oro__`方法），并提供了`pipe`方法，为LCEL的 `|` 语法提供了支持。

`Runnable`只定义了一个`name`属性，用于标识Runnable对象的名称。

`Runnable`定义了如下常用的接口方法:
  - `invoke`/`ainvoke`: 输入单条，输出结果
  - `batch`/`abatch`: 批量invoke，输出结果
  - `stream`/`astream`: 流式调用invoke
  - `batch_as_completed`/`abatch_as_completed`: 批量invoke直到完成

> 上述所有方法中，只有 `invoke` 方法是抽象方法，其他方法都有默认实现，所以如果要继承 `Runnable` 时，必须要实现的方法只有 `invoke`。

此外，`Runnable`还定义了如下几个接口方法，它们均返回`RunnableBinding`对象，对当前Runnable对象进行一些封装并附加一些参数/属性：
- `bind`: 以关键字参数附加一些参数/属性
- `with_config`: 以`RunnableConfig` + 关键字参数附加信息
- `with_listeners`/`with_alisteners`: 给Runnable对象，添加一些监听器，在运行开始，运行完成时，运行出错后，调用对应的监听回调函数。
- `with_types`:
- `with_retry`:
- `with_fallbacks`:
- `as_tool`:

上面的`with_listeners`/`with_alisteners`方法接受的Callable对象签名是：`Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]`


**`RunnableSerializable`**

继承自 `load`模块的`Serializable` + `Runnable`，是大部分LLM/ChatLLM的基类，**注意，它不是抽象类，不过一般不会直接使用**。

`RunnableSerializable`定义了如下两个在运行修改配置的接口方法：
- `configurable_fields`: 
- `configurable_alternatives`: 


上面两个类是整个`runnable`模块的基础，除此之外，`base.py`文件里，还提供了一些Runnable的常用封装类，方便使用，列举如下：
- `RunnableLambda`: 用于将任意Callable对象封装成Runnable对象，很常用
- `RunnableSequence`: 组合多个Runnable对象，LCEL语法的 `|` 运算符返回的就是这个对象，也很常用
- `RunnableBinding`: 对Runnable对象进行封装并附加一些参数/属性，**相当于 Runnable 装饰器**，LangChain框架内部很多地方都用到了它。
- `RunnableParallel`: 用于并行执行多个 Runnable 对象。
  它将输入数据分发给多个独立的处理步骤，并将它们的结果合并为一个输出字典。
- `RunnableGenerator`:
- `RunnableEach`:



#### `config.py`和`configurable.py`

`config.py`模块定义了`RunnableConfig`类——实际上就是一个Dict对象，用于封装运行时参数，可封装的参数如下：
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

- `base.py`: 定义了回调函数的 Mixin 类，回调函数通过 callback handler 定义
  - 一系列Mixin类：`RetrieverManagerMixin`, `LLMManagerMixin`, `ChainManagerMixin`, `ToolManagerMixin`, `CallbackManagerMixin`等
  - `BaseCallbackHandler`: 同步回调函数handler的接口类
  - `AsyncCallbackHandler`: 异步回调函数handler的接口类
  - `BaseCallbackManager`: 定义了回调函数管理器的基础类——它不是抽象类

- `manager.py`: 定义了一系列回调管理器，常用的有如下两个：
  - `CallbackManager`: 同步callback handler管理器，继承自`BaseCallbackManager`
  - `AsyncCallbackManager`: 异步callback handler管理器，继承自`BaseCallbackManager`

- `file.py`: 定义了一个`FileCallbackHandler`供使用，继承自`BaseCallbackHandler`
- `stdout.py`: 定义了一个`StdOutCallbackHandler`供使用，继承自`BaseCallbackHandler`
- `streaming_stdout.py`: 定义了一个`StreamingStdOutCallbackHandler`供使用，继承自`BaseCallbackHandler`


**使用说明**

- `base.py`中定义了一系列的Mixin类，它们定义了各个组件的事件方法，比如`on_llm_start`/`on_chat_model_start`/`on_chain_start`等方法。

- `BaseCallbackHandler`组合了上述Mixin类，是所有CallbackHandler（包括`AsyncCallbackHandler`）的基类。    
  但需要注意的是，虽然`BaseCallbackHandler`/`AsyncCallbackHandler`不是抽象类，里面的所有方法都是空的，
  所以实际使用时，需要继承此类，并实现自己需要的方法。

- `BaseCallbackManager`是回调管理器的基类，它定义并实现了一些基础方法，
不过一般不需要直接使用此类，而是使用子类`CallbackManager`/`AsyncCallbackManager`。

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
  - `SimpleChatModel`，继承自`BaseChatModel`，自定义ChatModel时，应当继承此类。


**使用说明**

`BaseLanguageModel`是所有LLM的基类，它定义了如下常用属性：
- `metadata`:
- `tags`:
- `verbose: bool`: 是否输出详细日志
- `callbacks`: 回调函数设置，它是一个`Union[list[BaseCallbackHandler], BaseCallbackManager]`，既可以是回调函数列表，也可以是回调管理器。
- `custom_get_token_ids`:

`BaseLLM`/`BaseChatModel`继承自`BaseLanguageModel`，它新增了如下属性：
- `callback_manager`: `BaseCallbackManager`类型，回调管理器。   
  不过**这个属性和`BaseLanguageModel`的`callbacks`属性功能重复了，所以被标识为废弃的，建议使用`callback_manager`属性**。


通常使用时，需要关注的是 `BaseLLM` 和 `BastChatModel` 提供的一些方法，列举如下：
- `invoke`/`ainvoke`: 输入单条，输出结果
  - 由`langchain_core/runnables/base.py`的`Runnable`抽象类定义的抽象方法。
- `batch`/`abatch`: 批量invoke，输出结果
  - 也是由`Runnable`抽象类定义的抽象方法。
- `stream`/`astream`: 流式调用invoke
  - 也是由`Runnable`抽象类定义的抽象方法。
- `generate_prompt`/`agenerate_prompt`: 
  - 输入一批prompt，调用模型产生输出，一般不需要手动调用。
  - 由 `BaseLanguageModel` 定义的抽象方法
- `predict`/`apredict`: 输入**单条** raw text，调用模型，并以raw text返回结果
  - 由 `BaseLanguageModel` 定义的抽象方法
- `predict_messages`/`apredict_messages`: 输入`List[BaseMessage]`，调用模型，并以`BaseMessage`返回结果
  - 由 `BaseLanguageModel` 定义的抽象方法
- `generate`/`agenerate`: 调用模型产生输出
  - 由`BaseLLM`/`BaseChatModel`类实现的方法，比较底层，一般不需要手动调用

> `BaseLLM` 和 `BastChatModel` 也提供了 `__call__` 方法，支持Callable调用的方式，不过看源码里，这种Callable调用被标记为废弃，后续1.0版本可能会移除掉。

整个抽象层次的调用逻辑为：`Runnable`定义抽象方法 --调用--> `BaseLanguageModel`定义抽象方法 --调用--> `BaseLLM`/`BaseChatModel`实现方法。

因此在**实际使用过程中，需要关注的是：`invoke`/`ainvoke`、`batch`/`abatch`、`stream`/`astream` 这3对方法**。

如果想基于`BaseLLM`/`BastChatModel`实现自己的模型，或者想看具体模型的实现，需要重点关注的是模型实现类里的如下方法：
- `_llm_type`：property，用户返回模型的唯一标识，必须要实现
- `_generate`/`_agenerate`: 必须要实现的模型调用方法
- `_stream`/`_astream`: 可选方法


### `messages`模块

用于封装 prompts 和 chat conversations 中的信息。

主要是和 prompts 模块中的 ChatMessagePromptTemplate 和 ChatPromptTemplate 搭配使用。

> 此模块只在`langchain_core`模块中有，可以直接使用，不需要在`langchain`等模块中继承。

**内容如下**：

- `base.py`
  - `BaseMessage`，所有消息的基类——注意，**它并不是抽象类**。
  - `BaseMessageChunk`，所有消息块的基类，大致类似于 `List[BaseMessage]`，这个**也不是抽象类**。

> 实际上 `BaseMessage` 是 pydantic的`BaseModel`子类。

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

**使用说明**

`BaseMessage`/`BaseMessageChunk`两个类已经定义好了重要的属性和方法，其他的Message类大部分都是简单的封装。
- 属性：
  - `id`: 消息标识符
  - `name`: 消息名称，可选
  - `type`: 消息类型，`HumanMessage`/`SystemMessage`等子类会设置这个字段，用于区分不同的消息类型。
  - `content`: 消息内容——最重要的部分
  - `role`: 消息角色，这个字段只有`ChatMessage`中有，其他子类没有。
  - `model_config`:
  - `response_metadata`:
- 方法：
  - `text`: 获取消息内容(`BaseMessage.content`)，返回一个字符串。 
  - `pretty_repr(html: bool = False)`:
  - `pretty_print()`:

需要注意的是，`ChatMessage` 是通用 Message 封装类，有一个 `role` 属性，而 `SystemMessage`/`HumanMessage`等子类没有这个属性，这一点不知道为啥。


### `prompts`模块

主要内容有：
- `base.py`
  - `BasePromptTemplate`, 所有Completion prompt模板的基类，这是个**抽象类**，不能直接使用。

- `message.py`
  - `BaseMessagePromptTemplate`，所有ChatModel的 Message prompt模板的基类，这是个**抽象类**，不能直接使用。

- `string.py`
  - `StringPromptTemplate`，继承自`BasePromptTemplate`，也是个**抽象类**，不能直接使用。

- `prompt.py`
  - `PromptTemplate`, 继承自`StringPromptTemplate`，这个类是最基础的prompt模板，**适用于Completion任务（普通的LLM模型）**。

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

> 注意，`ChatMessagePromptTemplate`返回的是`ChatMessage`，有 role 属性，type属性是'chat';
> `HumanMessagePromptTemplate`返回的是`HumanMessage`，type属性是'human'，**没有 role 属性**。

**使用说明**



### `prompt_values.py`

定义了 Prompt Template 的输出值。

内容如下：
- `PromptValue`: 封装了 prompt 的输出值，这个类是一个**抽象类**
  - 它继承自`Serializable`——也是pydantic的`BaseModel`子类，它也是下面所有类的基类。
  - 主要定义了两个方法：`def to_messages(self) -> list[BaseMessage]` 和 `def to_string(self) -> str`

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
## 数据检索相关

构建LEDVR工作流相关的模块:
- Loader: 加载器，用于加载Document数据
- Embedding: 向量嵌入，生成Document的Text Embedding向量
- Documentation Transform: 对Document进行转换，生成新的Document
- VectorStore: 向量数据库，用于存储Document的Text Embedding向量
- Retriever: 向量检索器，统一封装VectorStore的检索功能


### `document_loaders`模块

> 还有一个 `load` 模块，该模块提供了序列化和反序列化相关的工具.

module内容如下：
- `base.py`
  - `BaseLoader`: 所有 Loader 的抽象基类——定义了统一接口
  - `BaseBlobParser`: 
- `blob_loaders.py`
  - `BlobLoader`
- `langsmith.py`
  - `LangSmithLoader`


**使用说明**

`BaseLoader`里定义了如下接口：
- `load`/`aload`: 加载数据，返回 `List[Document]`
- `lazy_load`/`alazy_load`: 迭代加载数据，返回 `Iterator[Document]`
- `load_and_split`: 加载并分割数据，返回 `List[Document]`

这几个方法也是所有Loader的通用方法。

### `documents`模块

module内容如下：
- `base.py`
  - `BaseMedia`: 所有 Media 的抽象基类，Media 包括text
  - `Blob`
  - `Document`: 文档的基类，包含text和metadata —— KEY
- `compressor.py`
  - `BaseDocumentCompressor`
- `transformers.py`
  - `BaseDocumentTransformer`

**使用说明**

`BaseMedia`继承自 `Serializable`，所以也是一个Pydantic的`BaseModel`子类，里面定义了如下两个属性：
- `id`: 可选str，用于标识文档
- `metadata`: dict，用于存储文档的元数据

`Document`继承自 `BaseMedia`，新增如下两个属性：
- `type`: 固定是 Document
- `page_content`: str，文档的文本内容


### `embeddings`模块

module内容如下：
- `embeddings.py`: 只有一个 `Embeddings` 抽象基类

**使用说明**   

`Embeddings`是一个元类，定义了如下方法：
- `embed_query`/`aembed_query`: 用于计算query的embedding向量，返回一个 `List[float]`
- `embed_documents`/`aembed_documents`: 用于**批量计算**query的embedding向量，返回一个 `List[List[float]]`


### `vectorstores`模块

module内容如下：
- `base.py`
  - `VectorStore`: 所有 VectorStore 的抽象基类
  - `VectorStoreRetriever`: 所有 VectorStoreRetriever 的抽象基类，它继承自 `retrievers.py`里的`BaseRetriever`
- `in_memory.py`
  - `InMemoryVectorStore`
- `utils.py`


**使用说明**   

`VectorStore`抽象基类里定义了如下方法：
- `add_texts`/`aadd_texts`
- `add_documents`/`aadd_documents`
- `delete`/`adelete`
- `get_by_ids`/`aget_by_ids`
- `search`/`asearch`
- `similarity_search`/`asimilarity_search`
- `similarity_search`/`asimilarity_search`
- `as_retriever`: 返回一个 `VectorStoreRetriever` 对象，这个方法比较实用 —— KEY


`VectorStoreRetriever`抽象基类里定义了如下方法：
- `add_documents`/`aadd_documents`
- `add_documents`/`aadd_documents`


### `retriever.py`

module内容如下：
- `BaseRetriever`类

---------------------------------------------------
## Memory相关

从langchain v0.3.3 版本开始，memory模块被表示为废弃。  

官方文档[How to migrate to LangGraph memory](https://python.langchain.com/docs/versions/migrating_memory/)建议转向使用 LangGraph.

根据上面的官方文档，Langchain 里有关 Memory 的设计思路经历了3个阶段：
1. 基于`BaseMemory`的早期设计
2. 基于 `RunnableWithMessageHistory` 或 `BaseChatMessageHistory` 的设计，这个设计思路还在沿用，适用于简单的场景
3. 基于 LangGraph 的思路，这个是后续的发展方向

`BaseChatMessageHistory` 是和 `langchain.memory` 模块的 `ChatBaseMemory` 配合使用的，大致流程是 `ChatBaseMemory` 会
将历史聊天记录的存储委托给某个 `BaseChatMessageHistory` 实现类来进行。

`RunnableWithMessageHistory` 的使用方式不一样，它是**为了和 LangGraph 配合使用，并且支持 LCEL 表达式**。

LangGraph支持多用户的聊天记录管理，也支持容错恢复功能。


### `memory.py`

只提供了一个类：`BaseMemory`，所有memory的基类，提供了一些通用的接口。   

`BaseMemory`继承了`Serializable`，所以也是一个Pydantic的`BaseModel`子类。

`BaseMemory`定义了如下抽象方法：
- `memory_variables`: 返回`list[str]`，表示此memory提供了哪些key给模型使用。
- `load_memory_variables`/`aload_memory_variables`: 返回一个字典
- `save_context`/`asave_context`: 保存上下文的输入和输出信息
- `clear`/`aclear`: 清空上下文信息


### `chat_history.py`

内容如下：
- `BaseChatMessageHistory`: 用于表示聊天历史记录的抽象基类
- `InMemoryChatMessageHistory`: 存放在内存中的聊天历史记录简单实现类


`BaseChatMessageHistory`定义了一个属性`messages: list[BaseMessage]`，还定义了如下抽象方法：
- `add_message`: 用于添加消息
- `add_messages`/`aadd_messages`: 用于批量添加消息
- `add_user_message`/`add_ai_message`: 用于添加用户/AI消息
- `aget_messages`: 异步获取历史消息
- `clear`/`aclear`: 清空历史消息


`InMemoryChatMessageHistory`就是一个简单的基于内存列表实现历史记录实现类。

**使用说明**

`BaseChatMessageHistory`是配合`langchain.memory.chat_memory.py`里的`BaseChatMemory`一起使用的。

上面的`InMemoryChatMessageHistory`实现类其实就是`BaseChatMemory`里的`chat_memory`默认实现。

---------------------------------------------------
## Agent相关

### `tools`模块

module内容如下：
- `base.py`
  - `BaseTool`: 所有工具类的抽象基类，它继承了 `RunnableSerializable`，所以也是一个Pydantic的`BaseModel`子类。
  - `BaseToolkit`: 所有工具集类的抽象基类，它没有继承 `BaseTool`，不过继承了Pydantic的`BaseModel`。
- `simple.py`
  - `Tool`: 工具类，继承自 `BaseTool`
- `structured.py`
  - `StructuredTool`: 结构化工具类，继承自 `BaseTool`
- `convert.py`: 提供了`@tool`装饰器，用于将函数转换为工具类（`StructuredTool`对象或者`Tool`对象）。
- `render.py`
- `reriever.py`


**使用说明**    
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

langchain-core里的agents内容并没有太多，主要在langchain包里。




---------------------------------------------------

# LangChain

module名称为`langchain`，所有的模块可以分为如下6大类：

---------------------------------------------------
## Model IO

> `langchain`模块的`llm`和`chat_models`模块里都只是提供了模型定义、加载初始化的内容，
> 返回的模型都是 `langchain_core.language_model`模块里抽象类的子类，
> 所以如果要研究调用过程，应该看`langchain_core.language_model`模块里的源码。

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

这个模块写的稍微有点搞笑，就是把 `langchain_core.prompts`模块里的对应内容导入过来。


### `output_parsers`模块

这个模块也是把 `langchain_core.output_parsers` 里的内容导入过来。

---------------------------------------------------
## 数据增强

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
## Chain核心模块

> langchain_core 没有对应的chains模块，因为chains相关的核心接口/抽象类都在 `langchain_core.runnables` 中定义好了。
> **`chains` 模块才是 langchain 包的核心内容**。

### `chains`模块

module内容如下：
- `base.py`:
  - `Chain`: Chain组件的抽象基类，它继承了 `RunnableSerializable`，所以也是一个Pydantic的`BaseModel`子类。

`chains`模块提供了一系列的Chain组件实现类。

**使用说明**

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
## Memory 相关

### `memeory`模块

这个模块混合了早期 **基于`BaseMemory`实现** 思路的Memory组件和 **基于`BaseChatMessageHistory`实现** 思路的Memory组件。

module里 **基于`BaseMemory`实现**的（常用）内容如下：
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


**使用说明**

`BaseChatMemory`抽象类继承了`BaseMemory`类，并在此基础上定义了如下属性：
- `chat_memory: BaseChatMessageHistory`: 这个就是配合下面的`BaseChatMessageHistory`使用的，它的默认实现就是`InMemoryMessageHistory`。
- `output_key: Optional[str] = None`:
- `input_key: Optional[str] = None`:
- `return_messages: bool = False`:


**`BaseChatMemory` 底层会将历史消息的读写操作委托给 `BaseChatMessageHistory` 实现类**，具体过程是：
`save_context`方法里调用`chat_memory: BaseChatMessageHistory`属性对象的`add_messages`方法。


**基于`BaseChatMessageHistory`实现** 的组件在`langchain.memory.chat_message_history`模块里。
> 此模块其实只是一个简单导入的封装，实际是从 `langchain_community.chat_message_histories` 模块导入对应的类。

常用的ChatMessageHistory实现类如下：
- `ChatMessageHistory`: 这个其实就是 `langchain_core.chat_history.py` 里 `InMemoryChatMessageHistory` 的别名
- `FileChatMessageHistory`: 基于本地文件实现
- `RedisChatMessageHistory`
- `SQLChatMessageHistory`
- `ElasticsearchChatMessageHistory`


---------------------------------------------------
## Agent相关

> LangChain的 Agent 相关模块在 0.3 版本之后有较大改动，`langchain.agents`模块里内容是之前构建Agent的方式，已被标记为废弃的，
> 在 1.0.0 版本之前都会被保留，参加官方文档 [How to migrate from legacy LangChain agents to LangGraph](https://python.langchain.com/docs/how_to/migrate_agent/).
> 后续LangChain官方推荐转向使用 LangGraph 构建 Agent 应用。
> 因此这里就**不再详细介绍 agents 模块相关内容了**。

### `tools`模块

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
# LangGraph

首先要明确的是，LangGraph并不依赖LangChain-Core或者LangChain，
参考官方[FAQ -> Do I need to use LangChain to use LangGraph? What’s the difference?](https://langchain-ai.github.io/langgraph/concepts/faq/#do-i-need-to-use-langchain-to-use-langgraph-whats-the-difference)。

LangGraph更像是一个高度抽象的基于图的Agent调度框架，参考官方文档 [LangGraph Glossary](https://langchain-ai.github.io/langgraph/concepts/low_level/)
的说法，LangGraph的核心抽象 Graph 有如下3个概念：
- `State`: 图的状态，这是 Graph 的核心，本质上就是一个dict，不过通常用 `TypeDict` 类或者 Pydantic的`BaseModel`类表示。
- `Nodes`: Graph 的计算节点，本质上就是一个Python函数（更广泛一点就是一个Callable对象），可以封装各种逻辑，比如langchain里LLM/ChatModel/Chain的调用
- `Edges`: Graph 的边，用来连接节点，本质上也是一个Python函数，用于根据当前的`State`判断并返回下一个要执行的`Node`的名称

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
## `pregel`模块

此模块是LangGraph 的 Runtime 实现，它基于Google的Pregel算法，该算法专门用于大规模的并行图计算。

下面的 `CompiledGraph` 类就继承了此模块提供的 `Pregel` 类。

这个模块应该是 LangGraph 的核心实现，研究起来难度比较高。


---------------------------------------------------
## `constants.py`

LangGraph的常量字符串定义，这些字符串使用了`sys.intern()`函数驻留内存，避免重复创建。

常用常量如下：
- `START`
- `END`


## `config.py`

---------------------------------------------------
## `graph`模块

### `graph.py`

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
- 


### `state.py`

有状态图的表示，定义了如下3个类：

- `StateNodeSpec`: 有状态节点表示，继承自 `NamedTuple`，有如下属性：
  - `runnable: Runnable`:
  - `metadata`:
  - `ends`:
  - `input: Type[Any]`: 记录了输入，也就是状态
  - `retry_policy`:

- `StateGraph`: 继承自 `Graph`

- `CompiledStateGraph`: 继承自 `CompiledGraph`

**使用说明**

**`StateGraph`的初始化方法里需要传入一个表示状态的对象**，其他大部分方法都和`Graph`一样。


### `message.py`

定义了如下2个类：

- `MessagesState`: 继承自 `TypedDict`。    
  定义了一个 `message` 属性，类型是`list[AnyMessage]`，并用`Annotated`注解设置了一个reducer函数`add_messages`。

- `MessageGraph`: 继承自 `StateGraph`

还定义了一个常用的reducer函数：`add_messages`。



---------------------------------------------------
## `channels`模块



---------------------------------------------------
## `checkpoint`模块

此模块对应的是LangGraph里的短期记忆机制，只维护每次会话内的历史消息记录。

### `base`子模块

- 定义了`CheckpointTuple`，继承于`NamedTuple`，用来表示一个状态快照，有如下属性：
  - `config: RunnableConfig`
  - `checkpoint: Checkpoint`
  - `metadata: CheckpointMetadata`
  - `parent_config: Optional[RunnableConfig] = None`
  - `pending_writes: Optional[List[PendingWrite]] = None`


- 定义了`BaseCheckpointSaver`基类，用来保存和加载状态快照。


### `memory`子模块

实现了一个`InMemorySaver`，基于内存来保存checkpoint。

### `serde`子模块

定义序列化/反序列化相关内容。



---------------------------------------------------
## `store`模块

此模块对应于 LangGraph 的长期记忆机制，用于保存和加载长期记忆。

### `base`子模块

- `BaseStore`: 所有Store类的抽象基类，定义了如下方法：
- `put`/`aput`
- `get`/`aget`
- `list_namespaces`/`alist_namespaces`
- `search`/`asearch`
- `batch`/`abatch`
- `delete`/`adelete`

### `memory`子模块

定义了一个`InMemoryStore`，基于内存来保存长期记忆。


---------------------------------------------------
## `prebuilt`模块

这个模块提供了一些用于构建Agent的预制组件。

**`ToolNode`**

封装tool的节点。

**`tools_condition`**

封装 tool 的条件边，源码里内部逻辑比较简单，就是判断 state 里有没有 messages，并且messages 最后一条是不是 AIMessage，是就调用 Tool,
否则转向 END。


**`create_react_agent`**

用于快速创建一个React Agent。


此外，还定义了一些注解类型，用于在 tool 函数中访问图的状态和存储。
- `InjectedState`
- `InjectedStore`

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
## `managed`模块

