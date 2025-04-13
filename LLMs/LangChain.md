[TOC]

这里对 LangChain 的各个package进行简单总结。

# langchain-core

package名称为`langchain_core`，需要关注的有如下内容。  

大部分模块的说明可以在该模块的 `__init__.py` 文件中找到。

## `runnables`模块

这个模块是langchain_core模块的核心模块，基于 Runnable设计模式 和 LangChain Expression Language (LCEL) 定义了一系列的接口规范。

------
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

整个抽象层次的调用逻辑为：`Runnable`定义抽象方法 --调用--> `BaseLanguageModel`定义抽象方法 --调用--> `BaseLLM`/`BaseChatModel`实现方法。

因此在**实际使用过程中，需要关注的是：`invoke`/`ainvoke`、`batch`/`abatch`、`stream`/`astream` 这3对方法**。


### `prompts`模块

主要内容有：
- `base.py`
  - `BasePromptTemplate`, 所有prompt模板的基类，这是个**抽象类**，不能直接使用。

- `string.py`
  - `StringPromptTemplate`，继承自`BasePromptTemplate`，也是个**抽象类**，不能直接使用。

- `prompt.py`
  - `PromptTemplate`, 继承自`StringPromptTemplate`，这个类是最基础的prompt模板。

- `few_shot.py`
  - `FewShotPromptTemplate`
  - `FewShotChatMessagePromptTemplate`

- `few_shot_with_templates.py`
  - `FewShotPromptWithTemplates`

- `pipeline.py`
  - `PipelinePromptTemplate`

  
ChatModel使用的Prompt在`chat.py`中定义，它和上面基于`base.py`里的`BasePromptTemplate`不太一样。

主要类如下（缩进表示继承关系）：
- `BaseMessagePromptTemplate`: 大部分 Chat 相关 Template 的抽象基类。
  - `MessagesPlaceholder`: 占位符，用于在 prompt 中插入一个变量，这个变量是一个列表，列表中的每个元素都是一个 `BaseMessage` 对象。
  - `BaseStringMessagePromptTemplate`: 抽象类，以下才是常用的 Prompt Template 实现类。
    - `ChatMessagePromptTemplate`: 专门用于生成符合对话格式的消息（如用户消息、AI 回复、系统提示等）.
      - 主要用于生成**单个**对话消息模板
      - 返回的是 `ChatMessage` 对象
      - **通用**模板类，用于创建包含特定角色（如用户、AI或系统）的消息模板。它允许你指定消息的角色，并通过占位符动态插入变量内容。
    - `HumanMessagePromptTemplate`
      - 生成**单个**对话消息模板，专门为创建用户（人类）消息而设计的一个特化版本的模板类，返回的是 `HumanMessage` 对象
    - `AIMessagePromptTemplate`
    - `SystemMessagePromptTemplate`

> 注意，`ChatMessagePromptTemplate`返回的是`ChatMessage`，有 role 属性，type属性是'chat';
> `HumanMessagePromptTemplate`返回的是`HumanMessage`，type属性是'human'，**没有 role 属性**。

- `BaseChatPromptTemplate`: 继承`BasePromptTemplate`，也是抽象类
  - `ChatPromptTemplate`: 用于组合 多个ChatMessagePromptTemplate 或者其他类型的提示模板（例如文本提示模板），形成一个完整的对话上下文。
    - 持有一个 `messages` 属性，类型是`List[BaseMessage]` 


**使用说明**


### `messages`模块

用于封装 prompts 和 chat conversations中的信息。

主要是和 prompts 模块中的 ChatMessagePromptTemplate 和 ChatPromptTemplate 搭配使用。

> 此模块只在`langchain_core`模块中有，可以直接使用，不需要在`langchain`等模块中继承。

**内容如下**：

- `base.py`
  - `BaseMessage`，所有消息的基类——注意，**它并不是抽象类**。
  - `BaseMessageChunk`，所有消息块的基类，大致类似于 `List[BaseMessage]`，这个**也不是抽象类**。

> 实际上 `BaseMessage` 是 pydantic的`BaseModel`子类。

- `human.py`
  - `HumanMessage`，继承自`BaseMessage`
  - `HumanMessageChunk`，继承自`BaseMessageChunk`

- `system.py`
  - `SystemMessage`,
  - `SystemMessageChunk`

- `chat.py`
  - `ChatMessage`
  - `ChatMessageChunk`

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

------
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

------
## Memory相关
### `memory.py`

只提供了一个类：`BaseMemory`，所有memory的基类，提供了一些通用的接口。


------
## Agent相关



------
## 回调函数

### `tools`模块

------------

# `langchain`

module名称为`langchain`，所有的模块可以分为如下6大类：

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

------
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

------
## Chain 相关

### `chains`模块

------
## Memory 相关

### `memeory`模块

------
## Agent相关

### `agents`模块


------
## 回调函数

### `callbacks`模块





