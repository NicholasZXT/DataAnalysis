[TOC]

# LlamaIndex 介绍

LlamaIndex 主要分为3个产品：
- LlamaParse，云服务产品，提供文档解析，非开源，无法本地化部署
- LlamaAgents，构建 Agent Workflows 的产品
- LlamaIndex，开源的RAG产品

# LlamaIndex发行历史

LlamaIndex 的 Python SDK 发布于2023.2，第一个版本是0.4.5，此后的大版本号演变比较保守，小版本号更新频繁。

2025年首个版本为 0.12.9，至今（2026.3）还是 0.14.x ，大版本还没有过渡到 1.x 版本。

-------
# LlamaIndex架构研究

> 基于 LlamaIndex-Core v0.14.18 版本研究。

LlamaIndex的核心包是`llama_index.core`，源码内容大体上可以分为如下几类：
1. LLM相关
2. RAG相关
3. Agent + Workflow相关
4. 工具组件

------
## 基本组件抽象

`llama_index.core`里为整个框架定义了一些抽象基础，具体有如下内容：

### `schema.py`

基于Pydantic的`BaseModel`定义了一些基础组件的类和枚举常量。

比较重要的基类有：
- `BaseComponent(BaseModel)`：LlamaIndex里大部分组件的基类，基于pydantic提供了一些基础的序列化方法。
- `TransformComponent(BaseComponent, DispatcherSpanMixin)`: Transformation类操作的抽象基类，定义了 `__call__()` / `acall()` 为转换的方法。
- `BaseNode`：抽象类，定义了Node需要有的字段和一系列访问的抽象方法

常用组件类：
- `Node(BaseNode)`
- `TextNode(BaseNode)`
- `IndexNode(TextNode)`
- `Document(Node)`

枚举常量：
- `NodeRelationship`

### `base.llms`模块

（1）`base.py`定义了LLM的抽象基类 `class BaseLLM(BaseComponent, DispatcherSpanMixin)`：

此抽象类定义了如下3个属性：
- `model_config: ConfigDict`
- `callback_manager: CallbackManager`
- `rate_limiter: Optional[Any]`

定义了LLM类必须要实现的抽象方法（接口）如下：
- `chat()` / `achat()`
- `stream_chat()` / `astream_chat()`
- `complete()` / `acomplete()`
- `stream_complete()` / `astream_complete()`

**这些方法都是纯接口定义，没有任何实现逻辑，因此 `BaseLLM` 类不能直接继承使用，应当使用下面的`LLM`。**

（2）`types.py` 定义了LLM的输入输出schema类：
- `ChatMessage` / `ChatResponse`
- `CompletionResponse`

### `base.embeddings`模块

定义了如下两个Embedding的抽象基类：
- `class BaseEmbedding(TransformComponent, DispatcherSpanMixin)`
- `class BaseSparseEmbedding(BaseModel, DispatcherSpanMixin)`

### `base.response`模块

定义了返回结果的结构类（都是`dataclass`类）：
- `Response`
- `PydanticResponse`
- `StreamingResponse`
- `AsyncStreamingResponse`

------
## LLM组件

和LLM调用相关的组件。

### 模型封装

`llama_index.core.llm`模块里定义了如下内容。

（1）`llm.py` 定义了 `LLM(BaseLLM)`，对`BaseLLM`进行了一些增强：
- 增加了 `system_prompt`, `completion_to_prompt`, `messages_to_prompt`, `output_parser` 等属性
- 增加了 `stream()`/`astream()`, `predict()`/`apredict()`, `predict_and_call()`/`apredict_and_call()` 方法
- 定义了 `structured_predict()`/`astructured_predict()`, `stream_structured_predict()`/`astream_structured_predict()`, `as_structured_llm()` 和结构化输出相关的方法。

上述这些方法里新增了结合 `dispatcher` 组件的逻辑。

（2）`function_calling.py` 定义了 `class FunctionCallingLLM(LLM)`：
- 增加了 `chat_with_tools()`/`achat_with_tools`, `stream_chat_with_tools()`/`astream_chat_with_tools()` 等方法。

（3）`structured_llm.py` 定义了 `class StructuredLLM(LLM)`：
- 它有点特殊，虽然继承了`LLM`类，但是实际上是通过组合的方式实现的`BaseLLM`抽象方法：它持有一个`LLM`对象实例（初始化时传入）。
- 所有结构化输出方法的实现，实际上是转发给持有的`LLM`对象的方法，并对返回结果进行封装。

（4）`custom.py` 定义了 `class CustomLLM(LLM)`:

### Embedding

`llama_index.core.embeddings`模块定义了如下内容。

（1）`loading.py`

实现了一个`def load_embed_model(data: dict) -> BaseEmbedding` 函数，用于加载embedding模型。

（2）`multi_modal_base.py`

定义了`class MultiModalEmbedding(BaseEmbedding)`。

（3）`mock_embed_model.py`

（4）其他内容不重要。

### Prompt封装

`llama_index.core.prompts`模块里定义了如下内容。

（1）`base.py` 定义了提示词的模板类：
- `class BasePromptTemplate(BaseModel, ABC)`，提示词模板抽象类，定义了如下抽象方法：
  - `get_template`
  - `format`
  - `partial_format`
  - `format_messages`

- `class PromptTemplate(BasePromptTemplate)`

- `class ChatPromptTemplate(BasePromptTemplate)`

（2）`rich.py` 定义了 `class RichPromptTemplate(BasePromptTemplate)`

（3）`prompt_type.py` 定义了枚举类 `class PromptType(str, Enum)`

（4）`mixin.py` 定义了 `class PromptMixin(ABC)`

（5）多个提示词模板
- `default_prompts.py`
- `system.py`
- `chat_prompts.py`
- ...

### 工具调用

`llama_index.core.tools`模块里常用内容如下。

（1）`types.py` 定义了工具调用的相关Schema和抽象类。

- `ToolMetadata`，工具的元数据，是一个dataclass类：
```python
@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None
    fn_schema: Optional[Type[BaseModel]] = DefaultToolFnSchema
    return_direct: bool = False
```

- `ToolOutput`，工具输出 schema
```python
class ToolOutput(BaseModel):
    """Tool output."""
    blocks: List[ContentBlock]
    tool_name: str
    raw_input: Dict[str, Any]
    raw_output: Any
    is_error: bool = False
    _exception: Optional[Exception] = PrivateAttr(default=None)
```

- `class BaseTool(DispatcherSpanMixin)` 和 `class AsyncBaseTool(BaseTool)`，作为Tool的抽象基类，除了将 `__call__`方法定义为抽象方法，没啥有效信息。

（2）`calling.py` 定义了调用工具的辅助函数。
- `call_tool()` / `acall_tool()`
- `call_tool_with_selection()` / `acall_tool_with_selection()`

（3）`function_tool.py` 定义了 `class FunctionTool(AsyncBaseTool)`，用于将函数封装为工具，并进行调用 —— KEY.

（4）`retriever_tool.py` 定义了 `class RetrieverTool(AsyncBaseTool)`，用于将RAG的 Retriever 封装成工具调用。

（5）`query_engine.py` 定义了 `class QueryEngineTool(AsyncBaseTool)`，用于将RAG的 QueryEngine 封装成工具调用。

### 结构化输出

#### 基本介绍

LlamaIndex对于结构化输出的实现有两种方案：

1. 基于LLM的 Completion/Chat API + Prompt + OutputParser：在Prompt中要求模型按照指定schema输出，然后使用 OutputParser 解析。
2. 基于LLM的 FunctionCalling API + Prompt: 将 Pydantic 对象封装成一个 Tool，在提示词中要求模型使用此Tool，然后使用Pydantic对象进行解析。

**第 1 种方案更通用，适用于所有的Text-LLM；第 2 种方案更准确，但是要求LLM支持FunctionCalling**。

`LLM`类为结构化输出提供了高级API：
- `structured_predict()`/`astructured_predict()`；
- `stream_structured_predict()`/`astream_structured_predict()`；
- `as_structured_llm()`，这个方法会将当前LLM封装为`StructuredLLM`，但是底层还是调用的 `structured_predict()` 等API。

这些API底层会**自动根据模型的元数据，来判断具体使用哪种方式实现结构化输出**。

除了上面的高级API之外，LlamaIndex还提供了更加底层的结构化输出控制方式：
1. *Pydantic Programs*：基于Pydantic，使用 FunctionCalling API / Text Completion API + Output Parser，这个**使用最广泛**。
2. *Pre-defined Pydantic Program*：LlamaIndex预定义的一些模式
3. *Output Parsers*：基于 Completion API 自定义提示词+解析过程。

#### `program`模块

比较有用的源码内容如下：

（1）`llm_program.py` 定义了 `class LLMTextCompletionProgram(BasePydanticProgram[Model])`.

（2）`function_program.py` 定义了 `class FunctionCallingProgram(BasePydanticProgram[Model])`.

（3）`utils.py` 定义了一些工具函数：
- `get_program_for_llm()`

#### `parser`模块

Output Parser功能主要由 `llama_index.core.output_parsers` 模块提供。

`pydantic.py` 文件定义了 `class PydanticOutputParser(BaseOutputParser, Generic[Model])`。

------
## RAG组件

LlamaIndex官方文档[Building a RAG pipeline -> Introduction to RAG](https://developers.llamaindex.ai/python/framework/understanding/rag/)里将RAG的阶段划分成如下5个：
1. Loading
2. Indexing
3. Storing
4. Querying
5. Evaluating

------
### Loading

Loading阶段是读取文档资料，这个Stage里涉及到如下抽象组件。

#### Documents&Node

LlamaIndex里对文档的表示，参考 *基本组件抽象* -> `schema.py` 里的内容。

#### 文档读取

文档读取的Reader有两种类型：
1. 本地文件读取：使用`SimpleDirectoryReader`类
2. 其他数据源：使用DataConnectors组件，也就是继承 `BaseReader` 的插件。

主要由 `llama_index.core.readers` 模块提供相关组件，其中的内容如下：

（1）`base.py` 定义了`class BaseReader(ABC)`。

（2）`file`包实现了 `class SimpleDirectoryReader(BaseReader, ResourcesReaderMixin, FileSystemReaderMixin)`，用于读取本地文件，比较常用。

（3）`json.py` 定义了 `class JSONReader(BaseReader)`。

#### 文档转换

有两类组件：
1. Node Parser：用于对整个文档（Document）进行切分，形成Node —— 这是所有处理步骤的开始。
2. Ingestion Pipeline：对整个文档预处理管道的抽象封装，一般涉及到如下预处理步骤：
    - 文档切分：`llama_index.core.node_parser` 模块提供
    - 元数据提取：`llama_index.core.extractors` 模块提供
    - 文档/段落Embedding生成：由Embedding模块提供

一、**Node Parser**

主要由 `llama_index.core.node_parser` 模块提供相关组件，主要内容如下：

（1）`interface.py`。

定义了如下基类：
- `class NodeParser(TransformComponent, ABC)`
- `class TextSplitter(NodeParser)`
- `class MetadataAwareTextSplitter(TextSplitter)`

（2）`file`模块，其中实现了如下Node Parser：
- `class SimpleFileNodeParser(NodeParser)`
- `class MarkdownNodeParser(NodeParser)`
- `class JSONNodeParser(NodeParser)`
- `class HTMLNodeParser(NodeParser)`

（3）`text`模块，其中实现了如下文本（Document）分割类：
- `class TokenTextSplitter(MetadataAwareTextSplitter)`
- `class SentenceSplitter(MetadataAwareTextSplitter)`
- `class SentenceWindowNodeParser(NodeParser)`
- `class CodeSplitter(TextSplitter)`

二、**Extractor**

由 `llama_index.core.extractors` 模块提供，主要内容如下：

（1）`interface.py`，定义了 `class BaseExtractor(TransformComponent)` 抽象基类作为接口：
- `def process_nodes(nodes: Sequence[BaseNode], ...) -> List[BaseNode]`
- `async def aprocess_nodes(nodes: Sequence[BaseNode], ...) -> List[BaseNode]`
- `def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]`
- `async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]`

（2）`metadata_extractor.py` 提供了如下实现：
- `TitleExtractor(BaseExtractor)`
- `KeywordExtractor(BaseExtractor)`
- `SummaryExtractor(BaseExtractor)`
- `QuestionsAnsweredExtractor(BaseExtractor)`

（3）`document_text.py` 提供了 `DocumentContextExtractor(BaseExtractor)`。

注意：上面这些 Extractor 的具体实现类，**底层也是通过调用LLM实现的元数据抽取**，不是通过规则实现的，开销比较大。

三、**Ingestion Pipeline**

主要由 `llama_index.core.ingestion` 模块提供相关组件，主要内容如下：

（1）`pipeline.py`，定义了 `class IngestionPipeline(BaseModel)`.

（2）`cache.py`，定义了 `class IngestionCache(BaseModel)`。

（3）其他源码文件内容一般不直接使用。

------
### Storing

LlamaIndex里Index的存储后端抽象。

LlamaIndex **对Index需要存储的数据进行了更加细致的分类**（参见官方文档 [Componenet Guides > Storing](https://developers.llamaindex.ai/python/framework/module_guides/storing/)）：
- *Document stores*: 存储文档文本，也就是 `Document`/`Node` 对象。
- *Index stores*: 存储Index的元数据。
- *Vector stores*: 存储Embedding。
- *Property Graph stores*: 存储知识图谱
- *Chat Stores*: 存储对话历史

不过在实际应用中，上述5类数据可能不会都用到，比如知识图谱和对话历史就不一定用到，而Documents/Nodes和Vector数据一般在向量数据库也支持两者同时存放，不必分开。

LlamaIndex 对需要存储的数据类型的上述划分可能一开始看上去把问题复杂化了，但仔细想来，这样的理论划分还是有好处的。

LlamaIndex 和 Storing 相关的模块有如下 3 个:
- `llama_index.core.storage`
- `llama_index.core.vector_stores`
- `llama_index.core.graph_stores`

注意，**这3个模块也偏向底层抽象，只提供了基于内存的简单Store实现**，生产级别的Store实现都交给了第三方插件实现，比如Chroma、Milvus等向量数据库组件。

#### `storage`模块

`llama_index.core.storage` 模块主要有如下组件：

一、**外部接口**

Storage的使用上下文由 `storage_context.py`里的 `class StorageContext` 定义，它是一个dataclass类，作为容器使用，持有如下属性：

```python
@dataclass
class StorageContext:
    """
    Storage context.
    The storage context container is a utility container for storing nodes,
    indices, and vectors. It contains the following:
    - docstore: BaseDocumentStore
    - index_store: BaseIndexStore
    - graph_store: GraphStore
    - property_graph_store: PropertyGraphStore (lazily initialized)
    - vector_store: BasePydanticVectorStore
    """
    docstore: BaseDocumentStore
    index_store: BaseIndexStore
    vector_stores: Dict[str, SerializeAsAny[BasePydanticVectorStore]]
    graph_store: GraphStore
    property_graph_store: Optional[PropertyGraphStore] = None
```

每个属性对应于下面某个具体的Store实现类的实例对象。

此外还提供了如下的工具方法：

- `from_defaults()`，使用默认配置创建 `StorageContext` 对象，当然，此方法也接受参数用于自定义某个store的实例对象。
- `persist()`: 对Store进行持久化存储，它内部会调用所有store对象的`.persist()`方法进行持久化。
- `add_vector_store()`：新增一个自定义的Store对象。

总的来说，这个上下文对象的封装比较简单。


二、**内部实现**

定义了一些常用 Store 的实现组件。

（1）`kvstore`模块，K-V类型的Store。

**这个模块是下面 `index_store` 和 `docstore` 的基础**，因为下面两者的默认实现就是基于K-V映射方式存储的，并且是存放在内存中的。

- `class BaseKVStore(ABC)`抽象类，定义了CRUD操作抽象方法：`put()`/`aput()`, `get()`/`aget()`, `get_all()`/`aget_all()`, `delete()`/`adelete()` 等。
- `class BaseInMemoryKVStore(BaseKVStore)`，增加了 `persist()` 和 `from_persist_path()` 这两个抽象方法，对应保存/读取操作。
- `class MutableMappingKVStore(Generic[MutableMappingT], BaseKVStore)`: 基于 `BaseKVStore` 的泛型基类

一般不会直接使用这里的对象。

（2）`index_store`模块，定义了Index元数据的存储抽象和简单实现。

- `class BaseIndexStore(ABC)`: 抽象基类，定义了 `IndexStruct` 相关的CRUD操作。
- `class SimpleIndexStore(KVIndexStore)`: 基于内存的简单实现，**将实际存储操作委托给了内部持有的`BaseKVStore` 对象**。

（3）`docstore`模块，定义了 Document/Node 的存储抽象和简单实现。

- `class BaseDocumentStore(ABC)`: 抽象基类，定义了`Document`相关的CRUD操作。
- `class SimpleDocumentStore(KVDocumentStore)`: 基于内存的简单实现，**也将实际存储操作委托给了内部持有的`BaseKVStore` 对象**。

（4）`chat_store`模块，定义了聊天历史消息的存储抽象和简单实现。

- `class BaseChatStore(BaseComponent)`: 抽象基类，定义了 `ChatMessage`/`List[ChatMessage]` 相关的CRUD操作。
- `class SimpleChatStore(BaseChatStore)`: 基于内存的简单实现，不过它内部没有使用上面的 `BaskKVStore`。
#### `vector_store`模块

定义了Embedding向量存储的组件。

（1）`types.py` 里定义了抽象基类和一系列schema类：
- Schama定义类：`VectorStoreQueryMode`枚举类，`VectorStoreQueryResult`dataclass类
- `class VectorStore(Protocol)`：定义了VectorStore的**协议**
- `class BasePydanticVectorStore(BaseComponent, ABC)`：VectoreStore的抽象基类

上面的抽象组件定义的是**基于 `Node`/`Document` 级别的CRUD操作，以及 query 查询操作**。

（2）`simple.py`，实现了一个基于内存的简单Vector类 `class SimpleVectorStore(BasePydanticVectorStore)`

#### `graph_store`模块

定义了基于图结构的Store相关组件。

（1）`types.py` 定义了图存储的抽象基类和构成组件：
- `class GraphStore(Protocol)`：图存储组件的协议
- `class PropertyGraphStore(ABC)`：图存储的抽象基类
- 图节点类：
    - `class LabelledNode(BaseModel)`
    - `class EntityNode(LabelledNode)`
    - `class ChunkNode(LabelledNode)`
    - `class Relation(BaseModel)`
- `class LabelledPropertyGraph(BaseModel)`：一个基于内存的图存储简单实现。

（2）`simple.py` 定义了 `class SimpleGraphStore(GraphStore)`

（3）`simple_labelled.py` 定义了 `class SimplePropertyGraphStore(PropertyGraphStore)`

------
### Indexing - KEY

LlamaIndex官方文档 [Learn -> Building a RAG pipeline -> Indexing -> What is an Index?](https://developers.llamaindex.ai/python/framework/understanding/rag/indexing/#what-is-an-index) 对 *Index* 概念介绍如下：

> In LlamaIndex terms, an `Index` is a data structure composed of `Document` objects, designed to enable querying by an LLM. Your Index is designed to be complementary to your querying strategy.

根据官方文档和我浏览源码后的理解，这里的 *Index* 不仅仅是对文档语料的Embedding，它的含义更加广泛一点：
- Index 是一个管理所有文档Document对象的集合，支持文档对象的CRUD操作；
- Index有自己的元数据；
- 负责对 Document/Node进行 Embedding 转换；
- 负责和底层的存储组件（由Store负责）进行交互；
- 同时不同类型的Index也有着不同的检索方式，需要对接不同类型的检索器（Retriever）/Query Engine/Chat Engine。

可以认为 **Index 是连接 Store、检索器（Retriever）/Query Engine/Chat Engine 的桥梁**。

由于底层不同类型存储组件的查询检索方式不一样，因此 **Index + Store + Retriever/QueryEngine/ChatEngine 这三者的具体实现类都是配套的**。

比如基于向量存储的`VectoreStoreIndex`，就是和底层的`VectorStore`（比如`ChromaVectorStore`） + 对应的 `VectorIndexRetriever` 一起使用的。

Index 主要由 `llama_index.core.indices` 模块提供相关组件，同时还有一个 `llama_index.core.data_structs` 模块提供了一些类型定义。

#### 基本抽象定义-KEY

##### `BaseIndex`

`llama_index.core.indices.base.py` 定义了 `class BaseIndex(Generic[IS], ABC)`，它是所有 Index 的抽象（泛型）基类。

`BaseIndex`的重要属性如下：
```python
class BaseIndex(Generic[IS], ABC):
    def __init__(...):
        ...
        # Index 结构对象
        self._index_struct: 
            
        # 对接底层的存储上下文对象
        self._storage_context = storage_context or StorageContext.from_defaults()
        # 直接引用 存储上下文对象 里的各个类型的 Store
        self._docstore = self._storage_context.docstore
        self._vector_store = self._storage_context.vector_store
        self._graph_store = self._storage_context.graph_store
        
        # 转换器列表
        self._transformations: List[TransformComponent] = transformations
        
        # 回调管理器
        self._callback_manager = callback_manager or Settings.callback_manager
```
可以看出，`BaseIndex` 是一个使用 **组合** 模式的类，它所有的方法都是将实际的操作委托给持有的对象：
- **所有的Documents/Nodes存储以及CRUD操作都委托给了内部持有的 `StorageContext` 对象**。
- BaseIndex里没有定义 Embedding 的属性，因为不是所有类型的Index都需要进行Embedding操作，因此Embedding的属性放在了具体子类中。

`BaseIndex`定义了 3 类方法：

（1）**初始化方法（静态方法）**：

- `from_documents()`
- `build_index_from_nodes()`

当然也可以直接调用类的初始化方法。

（2）文档的**CRUD方法**：`insert_nodes()`/`ainsert_nodes()`, `delete_nodes()`/`adelete_nodes()`, `update_ref_doc()`/`aupdate_ref_doc()` 等

（3）**3个检索器生成方法**：
```python
class BaseIndex(Generic[IS], ABC):
    ...
    
    # 只有这个方法是抽象方法，必须实现
    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> BaseRetriever: ...
    
    # 返回一个 Query Engine，有默认实现
    def as_query_engine(
        self, llm: Optional[LLMType] = None, **kwargs: Any
    ) -> BaseQueryEngine:
        ...
        # 默认实现会调用 self.as_retriever() 方法
        
    # 返回一个 Chat Engine，有默认实现
    def as_chat_engine(
        self,
        chat_mode: ChatMode = ChatMode.BEST,
        llm: Optional[LLMType] = None,
        **kwargs: Any,
    ) -> BaseChatEngine:
        ...
        # 默认实现会调用 self.as_query_engine() 方法
```
查看源码可以发现，上面的3个对象是**递进的：`BaseRetriever -> BaseQueryEngine -> BaseChatEngine`，前面依次是后者的基础**，这也是为什么只有 `as_retriever()` 是抽象方法。

更重要的是，不同类型的 Index，检索的实现方式也是不一样的，比如基于向量数据库和基于知识图谱的Index的检索方式就不一样（甚至底层存储也不一样），    
这意味着 **Index 和 Retriever 是成对使用的，因此下面的具体实现类中也是成对定义的**。

##### `data_structs`模块

`llama_index.core.data_structs` 模块为 Index 提供了一些类型定义，内容如下。

（1）`data_structs.py` 里定义了各种 Index 类型对应结构的dataclass类：
- `class IndexStruct(DataClassJsonMixin)`：最基本的Index结构
- `class IndexGraph(IndexStruct)`
- `class KeywordTable(IndexStruct)`
- `class KG(IndexStruct)`
- 。。。

（2）`struct_type.py` 里定义了一个枚举类 `IndexStructType`，它的枚举常量就是所有的Index类型。

（3）`registry.py`里定义了一个 Dict 变量 `INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS`，保存的就是 `IndexStructType`枚举值 -> 具体IndexStruct的映射。

#### 具体实现类

常用的有如下几类：

（1）`vector_store`模块，定义了常用的向量索引 —— 最常用。

`class VectorStoreIndex(BaseIndex[IndexDict])` + `class VectorIndexRetriever(BaseRetriever)`.

（2）`knowledge_graph`模块，定义了基于知识图谱的Index —— 不过这个被标记为废弃了，推荐转向使用下面的 `property_graph` 包。

`class KnowledgeGraphIndex(BaseIndex[KG])` + `class KnowledgeGraphRAGRetriever(BaseRetriever)`

（3）`property_graph`模块，

`class PropertyGraphIndex(BaseIndex[IndexLPG])` + `class PGRetriever(BaseRetriever)`

（4）`tree`模块

`class TreeIndex(BaseIndex[IndexGraph])` + `class TreeRootRetriever(BaseRetriever)`/`class TreeSelectLeafRetriever(BaseRetriever)`

（5）`document_summary`模块

`class DocumentSummaryIndex(BaseIndex[IndexDocumentSummary])` + `class DocumentSummaryIndexLLMRetriever(BaseRetriever)`

------
### Querying

LlamaIndex的RAG查询功能相关组件。

RAG查询过程可以分为如下阶段：

1. 检索器（Retrieval），根据Query从IndexStore里检索获取 Documents/Nodes。
2. 后处理（Postprocessing），对检索得到的 Documents/Nodes 进行后处理，包括重排序，取TopK等操作。
3. 检索增强（Query Engine/Chat Engine），根据Query和检索后处理的Documents/Node，输入LLM。
4. 响应合成（Response Synthesis），对LLM返回的结果进行合并处理。

除了上面的主要步骤，还有其他一些附加功能：
- `Router`
- 结构化输出

#### 检索器

一、**抽象基类**

（1）`base.base_retriever.py`里定义了 `class BaseRetriever(PromptMixin, DispatcherSpanMixin)`.

（2）`base.base_auto_retriever.py`里定义了 `class BaseAutoRetriever(BaseRetriever)`.

（3）`base.base_multi_model_retriever.py`里定义了 `class MultiModalRetriever(BaseRetriever, BaseImageRetriever)`.

`BaseRetriever`抽象类定义了如下重要方法：

提供了如下两个对外接口用于检索 query：
- `def retrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]`
- `async def aretrieve(self, str_or_query_bundle: QueryType) -> List[NodeWithScore]`

上面两个接口内部依赖如下两个待实现的抽象方法：
- `def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]`
- `async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]`，内部其实是调用 `_retrieve()`。

二、**具体实现**

由于检索过程是依赖于底层的Index及Storage组件的，所以**不同类型的检索器具体实现类放在了 `llama_index.core.indices` 模块里**，和Index类型成对出现。

三、**检索策略**

`llama_index.core.retrievers`模块里定义的是**多个检索结果的合并策略**:

（1）`auto_merging_retriever.py` 里定义了 `class AutoMergingRetriever(BaseRetriever)`.

（2）`transform_retriever.py` 里定义了 `class TransformRetriever(BaseRetriever)`.

（3）`router_retriever.py` 里定义了 `class RouterRetriever(BaseRetriever)`.

（4）`recursive_retriever.py` 里定义了 `class RecursiveRetriever(BaseRetriever)`.

#### Router

参考官方文档说明 [LlamaIndex Framework -> Component Guides -> Querying -> Router](https://developers.llamaindex.ai/python/framework/module_guides/querying/router/).

Router的作用是根据用户的 query（及其元数据），在进行检索前：
- 决定选择哪个数据源的 Retriever及其Index
- 决定进行 Summarization 还是 Semantic Search
- 决定是否使用多个 Retriever，并如何合并它们的检索结果

一、**抽象基类**

`base.base_selector.py`里定义了如下Schema和抽象组件：

（1）Schema类：`class SingleSelection(BaseModel)`——单选；`class MultiSelection(BaseModel)`——多选
```python
class SingleSelection(BaseModel):
    """A single selection of a choice."""
    index: int
    reason: str
    
class MultiSelection(BaseModel):
    """A multi-selection of choices."""
    selections: List[SingleSelection]
```

（2）`class BaseSelector(PromptMixin, DispatcherSpanMixin)` 抽象基类：
- `select()` / `aselect()`: 对外API，返回 `SelectorResult` 对象——也就是 `MultiSelection` 对象；
- 抽象方法 `_select()` / `_aselect()` ，由上面调用。

二、**具体实现**

主要由 `llama_index.core.selectors` 模块提供，常用组件如下：

（1）`PydanticSingleSelector(BaseSelector)` / `PydanticMultiSelector(BaseSelector)`

（2）`LLMSingleSelector(BaseSelector)` / `LLMMultiSelector(BaseSelector)`

不论是上面哪种组件，底层都是通过请求LLM实现的：首先将每个Retriever封装成`RetrieverTool`，选择一个 selector， 然后使用专门的 `RouterRetriever` 来封装。

不同的selector使用LLM来选择工具的方式不一样，两者区别如下：

| 对比维度  | PydanticSingleSelector                                         | LLMSingleSelector                               |
|:------|:---------------------------------------------------------------|:------------------------------------------------|
| 底层技术  | 利用大模型的 函数调用 (Function Calling) 能力。                             | 利用大模型的 文本生成 (Text Completion) 能力。               |
| 工作原理  | 将每个候选项（如Retriever）的名称和描述构造成一个函数（Pydantic模型），让大模型通过调用函数的方式做出选择。 | 将所有候选项的文本描述放入一个提示词（Prompt）中，让大模型通过生成文本来决定选择哪一个。 |
| 性能与成本 | 更快，成本更低。因为它通常只需要一次高效的函数调用API请求。                                | 相对较慢，成本更高。因为它依赖于生成较长文本的API请求。                   |
| 决策能力  | 结构化，但灵活性稍弱。选择结果严格遵循预定义的函数模式，对于非常复杂的推理场景可能受限。                   | 更智能，灵活性高。可以利用大语言模型强大的文本理解和推理能力，处理更复杂、更微妙的选择逻辑。  |


总结：**PydanticSingleSelector 速度快、成本低，而 LLMSingleSelector 更智能、更灵活**。

三、**使用方式**

官方示例：
```python
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import RetrieverTool

# define indices
...

# define retrievers
vector_retriever = vector_index.as_retriever()
keyword_retriever = keyword_index.as_retriever()

# initialize tools: 将 Retriever 封装成 RetrieverTool
vector_tool = RetrieverTool.from_defaults(
    retriever=vector_retriever,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On.",
)
keyword_tool = RetrieverTool.from_defaults(
    retriever=keyword_retriever,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On (using entities mentioned in query)",
)

# define retriever
retriever = RouterRetriever(
    # 使用 selector
    selector=PydanticSingleSelector.from_defaults(llm=llm),
    retriever_tools=[
        list_tool,
        vector_tool,
    ],
)

# 这之后，可以将 RouterRetriever 视为基本的 Retriever 使用。
```

#### 后处理

由 `llama_index.core.postprocessor` 模块定义相关组件。

一、**抽象基类**

`types.py`里定义了如下抽象类：

```python
class BaseNodePostprocessor(BaseComponent, DispatcherSpanMixin, ABC):
    ...
    # 对外接口
    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        ...
    
    # 必须要实现的抽象方法
    @abstractmethod
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
```

二、**具体实现**

`postprocessor`模块里其他的源文件定义了如下比较常用的后处理实现类：
- `SimilarityPostprocessor`
- `KeywordNodePostprocessor`
- `DocumentWithRelevance`
- `SentenceTransformerRerank`
- `LLMRerank`
- `StructuredLLMRerank`

#### 响应合成

**负责将 原始query + 检索结果 合并，请求LLM，获取返回结果**。

主要由 `llama_index.core.response_synthesizers` 模块定义，主要组件如下。

##### 抽象基础

`base.py` 定义了 `class BaseSynthesizer(PromptMixin, DispatcherSpanMixin)`：
```python
class BaseSynthesizer(PromptMixin, DispatcherSpanMixin):
    """Response builder class."""

    def __init__(
        self,
        # 持有 LLM 对象
        llm: Optional[LLM] = None,
        # 回调函数管理器
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        # 是否流式返回
        streaming: bool = False,
        # 输出schema
        output_cls: Optional[Type[BaseModel]] = None,
        empty_response: Optional[str] = None,
    ) -> None:
        ...
        
    # ---------- 以下两个是待实现的抽象方法 ---------
    @abstractmethod
    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...

    @abstractmethod
    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get response."""
        ...
        
    # ---------- 对外接口 ----------
    # 它们会调用上面的 ger_response 方法
    def synthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        ...
    async def asynthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        ...
```

##### 具体实现

常用的有如下类：
- `SimpleSummarize`
- `TreeSummarize`
- `Refine`
- `CompactAndRefine`
- `Accumulate`

##### 使用方式

响应合成过程有多种模式，`llama_index.core.response_synthesizers.type.py` 里定义了一个 `ResponseMode(str, Enum)` 枚举类，还提供了一个 `get_response_synthesizer()` 函数，用于根据不同的 `ResponseMode` 获取预定义的响应合成实现类。

基本使用模式如下：
```python
from llama_index.core.data_structs import Node
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer

response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT
)

response = response_synthesizer.synthesize(
    "query text", nodes=[Node(text="text"), ...]
)
```

具体响应模式可以参考 [Configuring the Response Mode](https://developers.llamaindex.ai/python/framework/module_guides/querying/response_synthesizers/#configuring-the-response-mode)，常用的如下：
- `Response.NO_TEXT`: 仅使用 Retriever 检索相关 Nodes，返回一个空字符串作为响应，**不请求 LLM**.
- `Response.CONTEXT_ONLY`: 将候选Nodes文本拼接起来，然后作为响应返回，**不请求LLM**.
- `Response.GENERATION`: 不使用候选Nodes，**直接使用 query 请求LLM**。
- `Response.SIMPLE_SUMMARIZE`: 将候选Nodes文本拼接起来，截掉超出长度的部分，然后和 query 一起请求LLM。
- `Response.TREE_SUMMARIZE`:
- `Response.REFINE`: 迭代合成：先将query和第一个node内容请求LLM，获取答案后，第二次使用 query + answer[1] + node[2] 请求LLM，直到最后一个Node。**有多少个后续Node，就请求多少次，开销极大**。
- `Response.COMPACT`: 压缩迭代合成：类似于 Refine，但是每次请求时，会使用尽可能多的 nodes 填充context windows，因此请求数比 Refine 模式少一些。
- `Response.ACCUMULATE`: 成对请求：将 query 和候选nodes分别请求LLM，最终将所有LLM返回拼接到一起，作为返回。
- `Response.COMPACT_ACCUMULATE`: 压缩成对请求。


#### Query Engine

一、**抽象基类**

`base.base_query_engine.py`里定义了 `class BaseQueryEngine(PromptMixin, DispatcherSpanMixin)`抽象类。

主要有如下方法：

- `query()`/`aquery()`，主要的查询方法，内部会调用抽象方法 `_query()`/`_aquery()`
- `synthesize()` / `asynthesize()`，对查询结果进行合成

这个抽象类没有定义属性，也没有定义使用逻辑，单纯的一个接口类，因此具体使用逻辑要看子类的实现。

二、**具体实现**

主要由 `llama_index.core.query_engine` 模块提供实现，其中常用的实现类如下：
- `RetrieverQueryEngine`
- `TransformQueryEngine`
- `MultiStepQueryEngine`
- `KnowledgeGraphQueryEngine`
- `CustomQueryEngine`

以默认的 `RetrieverQueryEngine` 为例，它的初始化方法如下：
```python
class RetrieverQueryEngine(BaseQueryEngine):
    ...
    def __init__(
        self,
        # 底层的 Retriever
        retriever: BaseRetriever,
        # 响应结果合成类
        response_synthesizer: Optional[BaseSynthesizer] = None,
        # 节点后处理步骤列表
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        ...
        
    # 比较常用的是如下的静态初始化方法 ------------------------ KEY
    @classmethod
    def from_args(
        cls,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.COMPACT,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        summary_template: Optional[BasePromptTemplate] = None,
        simple_template: Optional[BasePromptTemplate] = None,
        output_cls: Optional[Type[BaseModel]] = None,
        use_async: bool = False,
        streaming: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "RetrieverQueryEngine":
        ...
```

具体查询使用时，它有3对查询接口：

（1）`retrieve()` / `aretrieve()`: 
1. 调用底层的 `Retriever.retrieve(query)` 获取查询结果 `List[NodeWithScore]`；
2. 逐个调用 `node_postprocessor.postprocess_nodes()` 方法；
3. 最后返回 `List[NodeWithScore]`

（2）`synthesize()` / `asynthesize()`: 
1. 调用内部的 `BaseSynthesizer.synthesize(query, List[NodeWithScore])` 方法，结合query和`retrieve()`的查询后处理结果进行合成；
2. 返回类型视具体的 `BaseSynthesizer` 实现类而定。

（3）`query()` / `aquery()`: 结合了上述两者，先调用 `retrieve()`，再调用 `synthesize()`。

三、**其他实现**

`llama_index.core.indices.struct_store` 模块里也提供了一些 `BaseQueryEngine` 的实现类：

------
## Chat Engine

ChatEngine 是用于对话流的组件，它底层实现依赖 Retriever 和 Memory，但是和 QueryEngine 不同，**不涉及后处理和响应合成**，因此没有这两个组件。

由 `chat_engine` 模块定义相关组件。

一、**抽象基类**

`types.py` 定义了抽象基类和相关Schema类：

（1）`class ChatMode(str, Enum)` 枚举类，表示聊天的类型，值如下：
- `SIMPLE`
- `CONTEXT`
- `REACT`
- `BEST`
- `CONDENSE_PLUS_CONTEXT`
- `OPENAI`

（2）`class ChatResponseMode(str, Enum)` 枚举类，只有如下两个值：
- `WAIT`
- `STREAM`

（3）`class AgentChatResponse` 和 `class StreamingAgentChatResponse` 返回结果的Schema类（dataclass数据类）。

（4）`class BaseChatEngine(DispatcherSpanMixin, ABC)`：**基础抽象类**

`BaseChatEngine(DispatcherSpanMixin, ABC)`定义了如下抽象方法：
- `chat()` / `achat()`
- `stream_chat()` / `astream_chat()`

但是它没有定义具体实现逻辑，因此还需要看具体子类实现。

二、**具体实现**

（1）`simple.py`里定义了 `class SimpleChatEngine(BaseChatEngine)`

（2）`context.py`里定义了 `class ContextChatEngine(BaseChatEngine)`


------
## Agent组件

主要由 `llama_index.core.agent` 模块提供，该模块内容主要分为两个部分：

（1）`react`模块，主要提供了 ReAct 模式Agent的一些辅助类，**并未提供实现**。

- `templates`文件夹里存放了一个`system_header_template.md`文本，定义的是 ReAct 模式Agent的提示词模板。
- `prompts.py` 读取上面的 `system_header_template.md` 并存入 `REACT_CHAT_SYSTEM_HEADER` 变量
- `formatter.py` 定义了`ReActChatFormatter` 类，用于将用户query合并到 `REACT_CHAT_SYSTEM_HEADER` 存放的 ReAct提示词里。
- `output_parser.py` 定义了 `ReActOutputParser` 类，用于解析 ReAct Agent 的输出
- `types.py` 定义了 `ActionReasoningStep`、`ObservationReasoningStep`、`ResponseReasoningStep` 等几个封装 ReAct 步骤的schema。

（2）`workflow`模块，**定义了Agent相关的抽象类和实现类**（但是不知道为啥取了`workflow`这个名字）：

- `prompts.py` 定义了几个提示词模板
- `workflow_events.py` 定义了各类 Event 的具体 schema 类，比如 `AgentInput`, `AgentStream`, `AgentOutput`, `ToolCall`, `ToolCallResult`, `AgentWorkflowStartEvent`。
- `agent_context.py` 定义了 `AgentContext` Protocol 类和一个简单的 `SimpleAgentContext` —— 它是一个 dataclass。
- **`base_agent.py` 定义了 `BaseWorkflowAgent` 抽象基类** —— 它继承自 `Workflow` 元类。
- `function_agent.py` 定义了 `class FunctionAgent(BaseWorkflowAgent)`
- `react_agent.py` 定义了 `class ReActAgent(BaseWorkflowAgent)`
- `codeact_agent.py` 定义了 `class CodeActAgent(BaseWorkflowAgent)`
- `multi_agent_workflow.py` 定义了 `AgentWorkflow`，用于封装多Agent协作。

>查看源码可以发现，`llama-index-core`里的 `agents` 模块底层**依赖的是 `llama-index-workflows` 包提供的抽象**：`llama_index.core.agent.workflow`会从`llama_index.core.workflow` 模块里导入内容，而它只是简单的从`workflow`（`llama-index-workflows`的包名）里导入对应的模块和对象而已。

此模块提供的`FunctionAgent`/`ReActAgent`/`CodeActAgent`是对 `llama-index-workflow` 里 `Workflow`类 的高层封装，使用起来比较简单，和`Workflow`的用法类似。

但是我看完Llama-Agent源码的感受是：**真要开发Agent，还是尽量避免使用Llama-Agent框架吧**。

------
## Evaluating

RAG、Agent效果评估组件，主要由 `llama_index.core.evaluation` 模块提供。

------
## 可观测性

LlamaIndex 提供的用于调试和跟踪问题的相关组件，主要有两类：
1. Callback Handler
2. Instrumentation

**回调函数是早期跟踪状态的工具，Instrumentation 是新版（v0.10.20之后）引入的新组件，后续会代替回调函数**。

LlamaIndex 沿用了可观测性组件里的概念，使用 *Trace*（追踪）来代表**一次完整的“用户请求生命周期”**。

具体到LlamaIndex的RAG场景，用户提出一个问题，LlamaIndex内部执行过程为：
1. 解析用户输入。
2. 检索相关文档片段（可能调用向量数据库）。
3. 构建 Prompt。
4. 调用 LLM API（可能多次，如果是 Agent）。
5. 执行工具（如运行 Python 代码）。
6. 合成最终答案。
7. 返回结果给用户。

这一整条从头到尾的执行链条，就是一个 Trace。

Trace 是树状结构的：
- 根节点 (Root)：代表整个请求的开始和结束。
- 子节点 (Spans)：代表过程中的具体步骤（如一次检索、一次 LLM 调用、一次工具执行）。

**Instrumentation是由 `llama-index-instrumentation` 包实现的**，`llama-index-core` 中虽然也有一个 `instrumentation` 模块，但是此模块大部分内容都是从`llama-index-instrumentation`导入的。

因此这里**只介绍 `llama-index-core` 包内置的 CallbackHandler 使用**。

Callback Handler 由 `llama_index.core.callbacks` 模块定义。

>看了下源码实现，Callback Handler的封装和使用逻辑在简单场景下，足够简单实用。
>但是性能开销比较大，而且是同步调用，在生产场景下（特别是异步和多线程场景），追踪链（Trace Context）容易丢失或错乱。

### Callback Handler抽象基础

（1）`base_handler.py` 定义了 `class BaseCallbackHandler(ABC)` 抽象基类。

此抽象基类定义了如下4个抽象方法：

```python
class BaseCallbackHandler(ABC):
    
    # --- Trace生命周期方法 ---
    # 当 LlamaIndex 开始处理一个新的顶层请求时调用。
    @abstractmethod
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""

    # 当整个请求处理完毕（无论成功还是失败），准备返回最终结果给用户时调用。
    @abstractmethod
    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
    
    # --- 特定事件处理 ---
    # 特定事件开始时执行，返回事件ID。
    @abstractmethod
    def on_event_start(
        self,
        event_type: CBEventType,  # 传入具体事件的枚举值
        payload: Optional[Dict[str, Any]] = None,  # 包含所有的信息
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""

    # 特定事件结束时执行，无返回值。
    @abstractmethod
    def on_event_end(
        self,
        event_type: CBEventType,  # 传入具体事件的枚举值
        payload: Optional[Dict[str, Any]] = None,  # 包含所有的信息
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
```

具体事件类型由 `CBEventType` 定义，并在`on_event_xxx()`方法中自己进行判断。

（2）`base.py` 定义了回调函数管理器 `CallbackManager(BaseCallbackHandler, ABC)` 和事件上下文 `EventContext`

回调函数管理器有如下属性：
- `self.handlers: List[BaseCallbackHandler]`: 回调函数列表
- `self._trace_map: Dict[str, List[str]]`

`CallbackManager`维护着一个回调处理器列表，并在特定事件发生时通知所有注册的处理器。

所有的 LlamaIndex 组件（如 QueryEngine, Agent, Workflow）内部都持有一个 CallbackManager 实例。

此外，`CallbackManager`提供了两个上下文接口（`@contextmanager`封装，配合`with`使用）：

- `as_trace(trace_id: str)`: 内部自动调用 `start_trace()` / `end_trace()` 方法，返回None
- `event(event_type: CBEventType, payload: Dict[str, Any], event_id: str)`: 内部自动封装事件类型为`EventContext`并返回， 
   会自动调用 `on_event_start()` / `on_event_end()` 方法

因此，LlamaIndex的源码中，`CallbackManager` 的使用方式一般为：

```python
with self.callback_manager.as_trace("query"):  # 设置当前trace ID
    with self.callback_manager.event(
        event_type=CBEventType.RETRIEVE,
        payload={},
    ) as event_context:  # 返回的是 EventContext 对象
        # 业务逻辑处理
        # 可以在业务逻辑处理中手动调用 EventContext 的 on_start() / on_end() 方法，
        # 不手动调用的话，event() 方法的with上下会也会自动调用，但不会重复调用
        ...
```

（3）`schema.py` 定义了常用模式：

- `class CBEventType(str, Enum)`: 回调事件的类型枚举类，定义了在哪些事件节点可以触发回调函数。

```python
class CBEventType(str, Enum):
    CHUNKING = "chunking"             # Logs for the before and after of text splitting.
    NODE_PARSING = "node_parsing"     # Logs for the documents and the nodes that they are parsed into.
    EMBEDDING = "embedding"           # Logs for the number of texts embedded.
    LLM = "llm"                       # Logs for the template and response of LLM calls.
    QUERY = "query"                   # Keeps track of the start and end of each query.
    RETRIEVE = "retrieve"             # Logs for the nodes retrieved for a query.
    SYNTHESIZE = "synthesize"         # Logs for the result for synthesize calls.
    TREE = "tree"                     # Logs for the summary and level of summaries generated.
    SUB_QUESTION = "sub_question"     # Logs for a generated sub question and answer.
    TEMPLATING = "templating"
    FUNCTION_CALL = "function_call"
    RERANKING = "reranking"
    EXCEPTION = "exception"
    AGENT_STEP = "agent_step"
```

- `class EventPayload(str, Enum)`：回调事件的Payload类型枚举类

- `class CBEvent`: Callback事件的dataclass类

```python
@dataclass
class CBEvent:
    """Generic class to store event information."""

    event_type: CBEventType
    payload: Optional[Dict[str, Any]] = None
    time: str = ""
    id_: str = ""

    def __post_init__(self) -> None:
        """Init time and id if needed."""
        if not self.time:
            self.time = datetime.now().strftime(TIMESTAMP_FORMAT)
        if not self.id_:
            self.id = str(uuid.uuid4())
```

- `class EventStats`: 事件统计的dataclass类

```python
@dataclass
class EventStats:
    """Time-based Statistics for events."""
    total_secs: float
    average_secs: float
    total_count: int
```

### Callback Handler具体实现

（1）`pythonically_priting_base_handler.py` 里定义了 `class PythonicallyPrintingBaseHandler(BaseCallbackHandler)` 类。

这个类比较简单，就是默认使用Python的`loggging`模块输出回调函数的日志，没有配置logger，才会使用`print()`方法。

（2）`class LlamaDebugHandler(PythonicallyPrintingBaseHandler)`，LllamaIndex内部使用的调试回调器，会记录所有类型的Event。

需要注意的是，它**只能在整个Trace执行结束后打印所有事件的日志，而不是在事件当时打印**。

（3）`class SimpleLLMHandler(PythonicallyPrintingBaseHandler)`

只处理`CBEventType.LLM`事件，会事件结束时打印LLM的输入和输出。

（4）`class TokenCountingHandler(PythonicallyPrintingBaseHandler)`，统计LLM和Embedding的Token使用情况。

------
# Instrumentation组件

> **v0.10.20**版本开始引入，新的 Instrumentation 组件是基于 OpenTelemetry (OTel) 和 OpenInference 标准重新开发的，复杂度比较高，但是更适合生产环境。

由单独的`llama-index-instrumentation`package定义，不过`llama-index-core`会依赖此package。

**以下总结内容基于 v0.5.0 版本源码**。

Instrumentation系统有如下 5 个核心组件：

- Event (事件)：是在 Span 生命周期中发生的特定动作的记录。
- EventHandler：事件处理器
- Span (跨度)：代表一个逻辑操作单元，类似于 OpenTelemetry 中的 Span 概念。它可以嵌套，形成树状的调用链路。
- SpanHandler (跨度处理器)：用户自定义的处理器，用于接收和处理 Span 事件。
- Dispatcher (事件调度器)：是 Instrumentation 系统的核心中枢，负责管理和分发所有的事件和 Span。

Event 和 Span 的区别在于：
- Event代表在 Span 生命周期内发生的特定瞬间的动作或数据快照。通常是一个时间点，没有层级结构。
- Span代表一个有持续时间的操作单元，一般对应于一个函数调用过程；具有 层级结构（树状），可以包含子 Span，也就是函数嵌套调用。

从源码里来看，两者比较独立，相互交互的情况很少，所以可以各自使用。

**Instrumentation的主要逻辑在两个地方**：

- `span_handlers` 模块定义的 `BaseSpanHandler` 及其子类实现中；
- `dispatcher.py` 文件定义的 `Dispatcher` 中。

其他模块（`base`, `events`, `span`, `event_handlers`）都是一些模型类定义和抽象类定义。

## `base`模块

（1）`event.py` 定义了事件对象的抽象基类：
```python
class BaseEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    id_: str = Field(default_factory=lambda: str(uuid4()))
    span_id: Optional[str] = Field(default_factory=active_span_id.get)  # type: ignore
    tags: Dict[str, Any] = Field(default={})
    ...
```

**所有的事件都必须继承此类**。

（2）`handler.py` 定义了：
```python
class BaseInstrumentationHandler(ABC):
    @classmethod
    @abstractmethod
    def init(cls) -> None:
        """Initialize the instrumentation handler."""
```
这个类也太抽象了。。。

## `event_handler`模块

（1）`base.py` 定义了抽象基类 `class BaseEventHandler(BaseModel)`:
```python
class BaseEventHandler(BaseModel):
    """Base callback handler that can be used to track event starts and ends."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "BaseEventHandler"

    # 只有这个抽象方法需要实现 ----------- KEY
    @abstractmethod
    def handle(self, event: BaseEvent, **kwargs: Any) -> Any:
        """Logic for handling event."""

    async def ahandle(self, event: BaseEvent, **kwargs: Any) -> Any:
        return self.handle(event, **kwargs)
```

**所有自定义Event事件（继承自`BaseEvent`）都需要编写该事件的Handler，两者是成对使用的**。

（2）`null.py`定义了`class NullEventHandler(BaseEventHandler)`，这也是一个空实现。

## `events`模块

定义了如下类：
```python
class SpanDropEvent(BaseEvent):

    err_str: str

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "SpanDropEvent"
```
这个Event类是框架内部使用的，专门用于调用抛出异常时，对Event进行Drop的事件。

## `span`模块

Span的特点：
- 每个 Span 有唯一的 ID 和父 Span ID（如果有）
- 可以记录开始时间、结束时间和持续时间
- 支持异步和同步操作
- 可以附加元数据和标签

Span的用途：追踪特定操作的执行过程，如一次检索、一次 LLM 调用等。

（1）`base.py` 定义了 
```python
class BaseSpan(BaseModel):
    """Base data class representing a span."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id_: str = Field(default_factory=lambda: str(uuid4()), description="Id of span.")
    parent_id: Optional[str] = Field(default=None, description="Id of parent span.")
    tags: Dict[str, Any] = Field(default={})
```

**所有的Span类型都必须继承此类**。

（2）`simple.py`定义了
```python
class SimpleSpan(BaseSpan):
    """Simple span class."""
    start_time: datetime = Field(default_factory=lambda: datetime.now())
    end_time: Optional[datetime] = Field(default=None)
    duration: float = Field(default=0.0, description="Duration of span in seconds.")
    metadata: Optional[Dict] = Field(default=None)
```

## `span_handler`模块 - KEY

作用：
- 接收 Span 的开始和结束事件
- 可以将数据发送到外部监控系统（如 Langfuse、Arize Phoenix、Datadog 等）
- 支持自定义逻辑进行日志记录、指标收集等

（1）`base.py`定义了抽象基类 `class BaseSpanHandler(BaseModel, Generic[T])`。

`BaseSpanHandler`里主要的方法如下：
- `span_enter()`，进入Span时调用，内部会调用**待实现的抽象方法`new_span()`**
- `span_exit()`，Span正常结束时调用，内部会调用**待实现的抽象方法`prepare_to_exit_span()`**
- `span_drop()`，Span内部发生异常时调用，内部会调用**待实现的抽象方法`prepare_to_drop_span()`**

`BaseSpanHandler`也是一个泛型类，它的泛型参数就是该SpanHandler对应要处理的 Span类型（`BaseSpan`子类）。

（2）具体实现类有 `class SimpleSpanHandler(BaseSpanHandler[SimpleSpan])` 和 `class NullSpanHandler(BaseSpanHandler[BaseSpan])`。

`NullSpanHandler`是一个空实现，被用作默认的处理器。

`SimpleSpanHandler`是一个简单的实现，只是创建一个 SimpleSpan 对象，并返回给调用方，背后没有对接 OTel 等可观测性系统。


## `dispatcher.py` - KEY

作用：
- 维护全局或局部的事件处理链
- 协调 Span 的生命周期管理
- 将事件分发给注册的处理器

其中定义了 `class Dispatcher(BaseModel)` 类和管理器 `class Manager`。

`Manager`类实现很简单，就是封装了一个`Dict[str, Dispatcher]`和对应的`add_dispatcher()`方法，重点是 `Dispatcher` 类。

`Dispatcher`类是整个观测流程的外部接口，它的主要方法可以分为3类：

（1）添加Handler:
- `add_event_handler()`
- `add_span_handler()`

（2）触发事件处理

事件处理的过程比较简单。

使用 `event(self, event: BaseEvent, **kwargs: Any)` / `aevent(self, event: BaseEvent, **kwargs: Any)` 触发某个Event对应的Handler。

传入的是 `BaseEvent`的某个子类，需要事先使用 `add_event_handler()` 方法添加能够处理此类型 Event 的处理器。

方法内部会遍历所有注册的EventHandler，并传入当前类型的Event，是否处理、如何处理全看EventHandler的`handle()`方法怎么实现的。

（3）触发Span处理

这部分稍微复杂些。

`Dispatcher`类实现了3个方法，用于触发`BaseSpanHandler`里对应阶段的方法：
- `span_enter()` -> `BaseSpanHandler.span_enter()`
- `span_exit()` -> `BaseSpanHandler.span_exit()`
- `span_drop()` -> `BaseSpanHandler.span_drop()`

上述3个方法里，**会遍历SpanHandler列表，调用每个SpanHandler的对应方法**，还会检查是否要调用父级Dispatcher的方法——Propagate行为。

手动调用上面Span各个阶段的方法很繁琐，还涉及到上下文管理，因此`Dispatcher`提供了一个重要方法 `span()`，**此方法是一个装饰器，用于装饰某个`Callable`对象，也就是该`Callable`对象的一次调用就被视为一个Span**。

`Dispatcher.span()`方法会在被装饰的`Callable`对象的执行前后调用上述生命周期方法，进行上下文管理以及异常处理等操作，支持同步/异步调用。

## 使用流程

整个模块的使用流程可以总结为如下：

（1）使用`llama_index_instrumentation.__init__.py` 里提供的 `get_dispatcher(name: str = "root") -> Dispatcher` 工具函数获取一个`Dispatcher`对象。
- `__init__.py` 里定义了一个 名为`root` 的 `root_dispatcher`，该 `Dispatcher` 注册了一个 `NullEventHandler` 和 `NullSpanHandler`；
- 创建 `root_manager: Manager = Manager(root_dispatcher)`对象，将`root_dispatcher` 添加到 `root_manager` 里，作为默认的Dispatcher；
- `get_dispatcher()` 函数会尝试根据传入的名称从 `root_manager` 里获取已有的 `Dispatcher`，或者是创建一个新的；
- 这个 `dispatcher` 对象通常是源文件/模块级别的“全局对象”。

（2）获得 dispatcher 对象后：
- 使用`@dispatcher.span`装饰器来装饰需要监控的方法/函数调用
- 在任意地方使用 `dispatcher.event()`/`dispatcher.aevent()` 方法触发对应事件

（3）对接OpenTelemetry
- llama-index-instrumentation 包并没有依赖或引入 OTel-API，它的默认实现是一个空实现，**不会向外部系统发送任何数据**。
- 如果需要对接OTel，需要安装`pip install llama-index-observability-otel`，该包提供了OTel的集成，包括`SpanHandler` 和 `EventHandler` 的OTel实现。
- 然后进行如下配置：

```python
# ------ 简单版本配置 ------
from llama_index.observability.otel import LlamaIndexOpenTelemetry
# initialize the instrumentation object
instrumentor = LlamaIndexOpenTelemetry()
if __name__ == "__main__":
    # start listening!
    instrumentor.start_registering()

# ------ 高级配置 ------
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from llama_index.observability.otel import LlamaIndexOpenTelemetry
# define a custom span exporter
span_exporter = OTLPSpanExporter("http://0.0.0.0:4318/v1/traces")
# initialize the instrumentation object
instrumentor = LlamaIndexOpenTelemetry(
    service_name_or_resource="my.test.service.1",
    span_exporter=span_exporter,
    debug=True,
)

if __name__ == "__main__":
    instrumentor.start_registering()
    # ... your code here
```



------
# Llama-Agent

官方文档 [Lllama Agents](https://developers.llamaindex.ai/python/llamaagents/overview/).

由**单独的package `llama-index-workflows` 提供**，不过这个package会被 `llama-index-core` 依赖，源码包路径为 `workflow`。

---
## Event-Based 模式

根据Llama-Agent官方文档的说法，和LangGraph那样基于DAG的Agent架构不一样，Llama-Agent的设计思路是 **基于Event + asyncio异步队列的生产者/消费者模式**。

> 个人感觉Llama-Agent相比LangGraph来说容易上手使用，但底层封装似乎比LangGraph还要深：
> 稍微看了下底层执行的源码，采用事件驱动 + asyncio的异步生产者/消费者的方式，但是**将Agent控制流逻辑 和 异步队列生产消费的逻辑 交叉耦合在一起**，看着就头大，感觉极难调试。

实际使用起来的模式如下（官方示例）：
```python
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent
# `pip install llama-index-llms-openai` if you don't already have it
from llama_index.llms.openai import OpenAI

# 1. 定义自己的事件类型，用于中间节点
class JokeEvent(Event):
    joke: str

# 2. 继承 Workflow 类
class JokeFlow(Workflow):
    # 持有一个LLM
    llm = OpenAI(model="gpt-4.1")

    # 3. 使用 @step 装饰 Workflow 中每一步的函数 ------------- KEY
    # 注意函数的 入参 和 返回值 类型
    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic

        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))

# 4. 执行
w = JokeFlow(timeout=60, verbose=False)
result = await w.run(topic="pirates")
print(str(result))
```

所谓的 **Event-Base**，指的是上面Workflow的构建流程中，并没有显式指定开始步骤、结束点、以及中间节点之间的步骤关系，整个workflow的流程的构建，主要依赖于`@step`装饰器。

`@step`装饰器会检查被装饰方法的函数签名，从中提取出入参类型和返回值类型——这些类型都必须是 `Event`及其子类：
- 入参为 `StartEvent`（及其子类）的，就是开始步骤
- 返回值为 `StopEvent`（及其子类）的，就是结束节点
- 中间步骤靠检查各个被装饰方法的 返回值类型 -> 下一个方法的入参 这样的逻辑串联起来，包括分支判断、并行、Human-in-the-Loop等逻辑，都依赖于此。

>个人感觉，这样的组装方式不是很好，一是严重依赖于被装饰函数的类型提示，二是感觉不如LangGraph那样显式构建节点和边的逻辑清楚。

## 源码研究

Llama-Agents的源码包名为`workflow`，其中有用的内容（基于 **V2.17.0** 版本）如下：

### `events.py`

定义了事件的schema `Event` 类及其子类，包括一些特殊事件schema，比如`StartEvent`, `StopEvent`, `InputRequiredEvent`, `HumanResponseEvent` 等。

### `context`模块

定义了Workflow的上下文抽象。

主要是`context.py`里的 `class Context(Generic[MODEL_T])` 类，它是上下文的基类，其他内容都是配合此上下文对象的，比如提供序列化/反序列化之类的。

### `decorators.py` - KEY

定义了一个高阶函数 `step()`，用于将方法/函数封装成 Workflow 中的一个Step（使用`StepFunction`表示）。

```python
def step(
    func: Callable[P, R] | None = None,
    *,
    workflow: type["Workflow"] | None = None,
    num_workers: int = 4,
    retry_policy: RetryPolicy | None = None,
    skip_graph_checks: list[StepGraphCheck] | None = None,
) -> Callable[[Callable[P, R]], StepFunction[P, R]] | StepFunction[P, R]:
```

此装饰器会解析被装饰方法/函数的签名，然后调用 `Workflow.add_step(func)`方法。

### `workflow.py` - KEY

定义了 `Workflow` 这个核心类。

#### 初始化方法

初始化方法签名如下：
```python
class Workflow(metaclass=WorkflowMeta):
    ...
    
    def __init__(
        self,
        # 超时时间设置
        timeout: float | None = 45.0,
        
        # 跳过 workflow 的校验流程
        disable_validation: bool = False,
        verbose: bool = False,
        
        # 用于依赖注入的自定义的资源管理器
        resource_manager: ResourceManager | None = None,
        # 限制 Workflow.run() 方法的并行调用次数
        num_concurrent_runs: int | None = None,
        runtime: Runtime | None = None,
        
        # Workflow 的名称
        workflow_name: str | None = None,
        skip_graph_checks: set[WorkflowGraphCheck] | None = None,
    ) -> None:
        ...
    
    ...
```

#### `run`方法

Workflow 的执行入口，这是一个异步方法：
```python
class Workflow(metaclass=WorkflowMeta):
    ...
    def run(
        self,
        # 传入的上下文
        ctx: Context | None = None,
        # 手动指定开始Event的类型
        start_event: StartEvent | None = None,
        **kwargs: Any,
    ) -> WorkflowHandler:
        ...
```
返回的 `WorkflowHandler` 是一个类似于 Future 的对象，可以用来 await 最终结果或者 stream events。

`run()` 方法里会做如下主要的操作：
1. 调用`self._validate()`，检查当前Workflow的定义；
2. 使用当前workflow对象来初始化一个 `Context` 变量 `ctx`；
3. 调用 `ctx._workflow_run()` --> `workflow._runtime.run_workflow()` --> `BasicRuntime.run_workflow()` 方法
4. `BasicRuntime.run_workflow()` 中，先调用 `self.register()` 方法将当前 Workflow 对象封装成 `RegisteredWorkflow` —— 这是一个简单的dataclass类
5. 封装 `RegisteredWorkflow` 对象时，会调用 `create_workflow_run_function()` 和 `as_step_worker_functions()` 这两个方法
6. `create_workflow_run_function()`方法内部会将Workflow对象作为函数 `run_workflow()` 的闭包，然后返回 `run_workflow` 函数本身，作为 `RegisteredWorkflow.workflow_run_fn` 属性值
7. `as_step_worker_functions()`方法会遍历 workflow 的steps，使用 `as_step_worker_function()` 将每个step中的函数封装为满足`StepWorkerFunction`协议的对象，然后返回一个 `dict[str, StepWorkerFunction]`，作为 `RegisteredWorkflow.steps` 属性值
8. await执行 `RegisteredWorkflow.workflow_run_fn` 函数，该函数内部会调用 `control_loop()` 函数（`runtime.types.step_fuction.py`）
9. `control_loop()` 函数是整个（异步）控制流的主逻辑 --> `_ControlLoopRunner.run()` 方法
10. **这部分逻辑代码可读性极差，逻辑太难梳理了**...

### `runtime`模块 - KEY

定义了 Workflow 运行时内部类和执行逻辑。

> 此模块的内容是 `Workflow.run()` 的底层实现，简单研究了下，个人感觉它**将 Agent控制流的逻辑 和 底层asyncio异步队列的消费者/生产者逻辑 深度耦合交织在一起**。   
> 我的评价是：**代码可读性极差，比LangChain/LangGraph还差**，绝对是个大坑！！！

### `representation`模块

定义了 Workflow 中各个step的抽象表示以及构建流程，主要内容如下。

（1）`types.py` 定义了step的抽象类 `class WorkflowNodeBase(BaseModel)` 及其子类：`WorkflowStepNode`, `WorkflowEventNode`, `WorkflowGenericNode`等

（2）`build.py` 定义了 `def get_workflow_representation(workflow: Workflow) -> WorkflowGraph` 函数及其辅助工具函数，用于将一个Workflow的step组装起来。

（3）`validate.py` 定义了用于校验 Workflow 相关的方法和类，比如 `build_step_graph()`, `validate_graph()` 等。

### `handler.py`

定义了 `class WorkflowHandler(Awaitable[RunResultT])`。
