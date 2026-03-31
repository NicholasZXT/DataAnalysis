"""
LlamaIndex使用研究
"""
from typing import List, Dict, Tuple, Any, Sequence
import os
from pathlib import Path
import asyncio
# %% ---------- LlamaIndex 核心包 ----------
# --- LlamaIndex 抽象基础 ---
from llama_index.core import Settings
# from llama_index.core.schema import BaseComponent, BaseNode
from llama_index.core.schema import NodeRelationship, Node, TextNode, ImageNode, IndexNode, Document, BaseNode
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response, StreamingResponse, AsyncStreamingResponse, PydanticResponse
# --- LLM 组件 ---
from llama_index.core.llms import (
    LLMMetadata, MockLLM, LLM, CustomLLM,
    MessageRole, ChatMessage, ChatResponse, CompletionResponse, TextBlock, DocumentBlock
)
from llama_index.core.prompts import Prompt, PromptType, PromptTemplate, ChatPromptTemplate, RichPromptTemplate
# --- RAG: Loading 组件 ---
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import (
    NodeParser, SimpleNodeParser, TokenTextSplitter, TextSplitter, SentenceSplitter, MarkdownNodeParser,
    HTMLNodeParser, CodeSplitter, SentenceWindowNodeParser, HierarchicalNodeParser
)
from llama_index.core.extractors import BaseExtractor, TitleExtractor, KeywordExtractor, SummaryExtractor, DocumentContextExtractor
from llama_index.core.ingestion import IngestionCache, IngestionPipeline, DocstoreStrategy, run_transformations, arun_transformations
# --- RAG: Storing 组件 ---
# from llama_index.core import StorageContext
from llama_index.core.storage import StorageContext
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.storage.docstore import SimpleDocumentStore, DocumentStore
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreInfo, VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.graph_stores import (
    SimpleGraphStore, PropertyGraphStore, SimplePropertyGraphStore, EntityNode, LabelledNode, ChunkNode, Relation
)
# --- RAG: Indexing 组件 ---
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices import (
    VectorStoreIndex, SummaryIndex, DocumentSummaryIndex, KeywordTableIndex, SimpleKeywordTableIndex, TreeIndex,
    KnowledgeGraphIndex, PandasIndex,
)
# --- RAG: Quering 组件 ---
from llama_index.core.retrievers import (
    VectorIndexRetriever, VectorIndexAutoRetriever, SummaryIndexRetriever,
    KGTableRetriever, KnowledgeGraphRAGRetriever,
    # BM25Retriever,
    RouterRetriever, TransformRetriever, QueryFusionRetriever, RecursiveRetriever, AutoMergingRetriever
)
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.selectors import (
    BaseSelector, SelectorResult, SingleSelection, MultiSelection,
    LLMSingleSelector, LLMMultiSelector, PydanticSingleSelector, PydanticMultiSelector, EmbeddingSingleSelector
)
from llama_index.core.postprocessor import (
    SimilarityPostprocessor, KeywordNodePostprocessor, DocumentWithRelevance, StructuredLLMRerank, LLMRerank
)
from llama_index.core.response_synthesizers import (
    BaseSynthesizer, Refine, SimpleSummarize, TreeSummarize, CompactAndRefine, Accumulate,
    ResponseMode, get_response_synthesizer
)
# --- Chat 相关组件 ---
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.chat_engine import SimpleChatEngine, ContextChatEngine, MultiModalContextChatEngine
from llama_index.core.memory import BaseMemory, Memory, ChatMemoryBuffer, ChatSummaryMemoryBuffer, VectorMemory
# --- Evaluating 组件 ---
# --- 可观测性 组件 ---
from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.callbacks import CallbackManager, CBEvent, CBEventType, LlamaDebugHandler, TokenCountingHandler
from llama_index_instrumentation import Dispatcher, get_dispatcher
from llama_index_instrumentation.base import BaseEvent, BaseInstrumentationHandler
from llama_index_instrumentation.event_handlers import BaseEventHandler
# --- RAG: Agent 组件 ---
from llama_index.core.agent import (
    BaseWorkflowAgent, AgentWorkflow, FunctionAgent, ReActAgent,
    AgentInput, AgentOutput, AgentStream, AgentStreamStructuredOutput,
    ToolCall, ToolCallResult
)
# --- Llama-Agent (workflow) 组件 ---
from workflows import Workflow, step, Context
from workflows.events import Event, StartEvent, StopEvent
from workflows.handler import WorkflowHandler

# %% ---------- LlamaIndex Hub 插件依赖 ----------
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import (
    CSVReader, DocxReader, PDFReader, MarkdownReader, PyMuPDFReader, UnstructuredReader
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# %% ---------- Llama LLM 使用 ----------
def llm_usage():
    """
    LLM使用
    """
    print("-------- Llama LLM Usage --------")
    llm = Ollama(
        model="qwen3.5:9b",
        base_url="http://localhost:11434",  # 默认值
        context_window=8192,
        request_timeout=120,  # thinking 开启时，这个要调大一点，避免 chat 方法超时
        is_function_calling_model=True,
        keep_alive="30m",
        thinking=False
    )

    prompt = "请介绍下你自己"
    print(">>> complete:")
    response: CompletionResponse = llm.complete(prompt=prompt)
    # print(response)
    print(response.text)
    print(response.additional_kwargs)
    print("\n")
    print(">>> stream_complete:")
    for chunk in llm.stream_complete(prompt=prompt):
        # print(type(chunk))  # 也是CompletionResponse
        # 在 stream 方式中，CompletionResponse.text 字段中，会不断更新，每次更新都会返回一个完整的结果
        # print(chunk.text)
        # 增量更新应当使用 delta字段
        print(chunk.delta, end="")
    print("\n")

    print("--------")
    msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content="你是一位机器学习专家"),
        ChatMessage(role=MessageRole.USER, content="请简单介绍下XGBoost算法的使用常见（200字以内）。")
    ]
    print(">>> chat:")
    response: ChatResponse = llm.chat(messages=msgs)
    # print(response)
    print(response.message)
    print(response.additional_kwargs)
    print("\n")
    print(">>> stream_chat:")
    for chunk in llm.stream_chat(messages=msgs):
        # print(type(chunk))  # 也是ChatResponse
        # message 字段也是全量信息
        # print(chunk.message)
        print(chunk.delta, end="")
    print("\n")

async def llm_usage_async():
    print("-------- Llama LLM Usage (Async) --------")
    llm = Ollama(model="qwen3.5:9b", request_timeout=120)

    prompt = "请介绍下你自己"
    print(">>> acomplete:")
    response: CompletionResponse = await llm.acomplete(prompt=prompt)
    # print(response)
    print(response.text)
    print(response.additional_kwargs)
    print("\n")
    print(">>> astream_complete:")
    async for chunk in await llm.astream_complete(prompt=prompt):
        print(chunk.delta, end="")
    print("\n")

    print("--------")
    msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content="你是一位机器学习专家"),
        ChatMessage(role=MessageRole.USER, content="请简单介绍下XGBoost算法的使用常见（200字以内）。")
    ]
    print(">>> achat:")
    response: ChatResponse = await llm.achat(messages=msgs)
    # print(response)
    print(response.message)
    print(response.additional_kwargs)
    print("\n")
    print(">>> astream_chat:")
    async for chunk in await llm.astream_chat(messages=msgs):
        print(chunk.delta, end="")
    print("\n")


# %% ---------- Llama Embedding 使用 ----------
def embedding_usage():
    """
    Embedding使用
    """
    print("-------- Llama Embedding Usage --------")
    embedding = OllamaEmbedding(
        model_name="bge-m3:567m",
        embed_batch_size=10
    )
    text = "数据挖掘"
    vector = embedding.get_query_embedding(query=text)
    print(len(vector))
    print(vector)

    vector = embedding.get_text_embedding(text=text)
    print(len(vector))
    print(vector)

    vector = embedding.get_general_text_embedding(texts=text)
    print(len(vector))
    print(vector)


# %% ---------- LLM/Embedding 获取 ----------
def get_llm() -> LLM:
    llm = Ollama(
        model="qwen3.5:9b",
        base_url="http://localhost:11434",  # 默认值
        context_window=8192,
        request_timeout=120,  # thinking 开启时，这个要调大一点，避免 chat 方法超时
        is_function_calling_model=True,
        keep_alive="30m",
        thinking=False
    )
    return llm

def get_embedding() -> BaseEmbedding:
    embedding = OllamaEmbedding(
        model_name="bge-m3:567m",
        embed_batch_size=10
    )
    return embedding


# %% ---------- Llama Prompt 使用 ----------
def prompt_usage():
    """
    Llama Prompt 使用。
    主要有如下几种方式：
    1. RichPromptTemplate: 新版的提示词模板，基于Jinja2模板创建
    2. PromptTemplate / ChatPromptTemplate : 旧版提示词模板
    """
    print("-------- Llama Prompt Usage --------")
    print(">>> PromptTemplate:")
    context_str = "具体的内容"
    query_str = "用户问题"
    template = (
        "以下是我们提供的上下文信息.\n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "请基于上述信息，回答问题: {query_str}\n"
    )
    print(">>>>>> prompt:")
    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context_str=context_str, query_str=query_str)
    print(prompt)
    print(">>>>>> messages:")
    messages = qa_template.format_messages(context_str=context_str, query_str=query_str)
    for msg in messages:
        print(msg)

    print(">>> ChatPromptTemplate:")
    message_templates = [
        ChatMessage(content="你是一位机器学习专家", role=MessageRole.SYSTEM),
        ChatMessage(content="请简单介绍下：{topic}", role=MessageRole.USER),
    ]
    print(">>>>>> prompt:")
    chat_template = ChatPromptTemplate(message_templates=message_templates)
    # or easily convert to text prompt (for completion API)
    prompt = chat_template.format(topic=...)
    print(prompt)
    print(">>>>>> messages:")
    # you can create message prompts (for chat API)
    messages = chat_template.format_messages(topic=...)
    for msg in messages:
        print(msg)

    print(">>> RichPromptTemplate:")
    # 注意，RichPromptTemplate 使用双括号（Jinja2语法），而不是单括号
    template = RichPromptTemplate(
    """
    以下是我们提供的上下文信息.
    ---------------------
    {{ context_str }}
    ---------------------
    请基于上述信息，回答问题: {{ query_str }}
    """
    )
    print(">>>>>> prompt:")
    # format as a string
    prompt_str = template.format(context_str=context_str, query_str=query_str)
    print(prompt_str)
    print(">>>>>> messages:")
    # format as a list of chat messages
    messages = template.format_messages(context_str=context_str, query_str=query_str)
    for msg in messages:
        print(msg)


# %% ---------- Llama RAG-Loading 使用 ----------
def rag_loading_usage():
    """
    RAG-Loading组件使用，包括如下内容：
    1. 文档读取：Reader读取本地文件；Data Connector读取其他数据源的文件
    2. 文档切分：NodeParser/TextSplitter
    3. 文档预处理Pipeline: Ingestion Pipeline，一般包含3个步骤：NodeParser/TextSplitter -> MetaData Extractor -> Embedding
    """
    print("-------- Llama RAG-Loading Usage --------")
    doc_dir = Path(r"C:\Users\Drivi\Documents\技术书籍\RAG-Documents")
    print("doc_dir: ", doc_dir, " exists: ", doc_dir.exists())
    files = list(doc_dir.glob("*.txt"))

    print(">>> Reader:")
    reader = SimpleDirectoryReader(input_files=files)
    for resource in reader.list_resources():
        print("resource: ", resource)
    docs: List[Document] = reader.load_data()
    for doc in docs:
        print("----------------")
        print("document:\n", doc)

    docs = docs[0:1]

    print("\n>>> NoderParser/TextSplitter:")
    # 主要有两个接口：get_nodes_from_documents 和 __call__ 方式（底层也是调用的get_nodes_from_documents）
    print(">>>>>> SentenceSplitter:")
    sent_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    # sent_nodes = sent_splitter.get_nodes_from_documents(docs)
    # 或者
    sent_nodes: List[BaseNode] = sent_splitter(docs)
    print("len(nodes): ", len(sent_nodes))
    for node in sent_nodes:
        print("----------------")
        print("node:\n", node)
        print("node.metadata: ", node.metadata)
    print(">>>>>> TokenTextSplitter:")
    token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=10, separator=" ")
    # token_nodes = token_splitter.get_nodes_from_documents(docs)
    token_nodes: List[BaseNode] = token_splitter(docs)
    print("len(nodes): ", len(token_nodes))
    for node in token_nodes:
        print("----------------")
        print("node:\n", node)
        print("node.metadata: ", node.metadata)

    print("\n>>> MetadataExtractor:")
    # 元数据提取也是通过调用大模型实现的，因此需要提供一个LLM示例，不提供的话，会使用 Settings.llm 这个全局默认LLM
    llm = get_llm()
    # 主要使用接口：process_nodes; __call__;
    print(">>>>>> TitleExtractor:")
    title_extractor = TitleExtractor(llm=llm)
    title_nodes = title_extractor(docs)
    print("len(nodes): ", len(title_nodes))
    for node in title_nodes:
        print("----------------")
        print("node:\n", node)
        # 抽取的信息存放在 metadata 字典的 document_title 字段
        print("node.metadata: ", node.metadata)
    print(">>>>>> KeywordExtractor:")
    keyword_extractor = KeywordExtractor(llm=llm)
    keyword_nodes = keyword_extractor(docs)
    print("len(nodes): ", len(keyword_nodes))
    for node in keyword_nodes:
        print("----------------")
        print("node:\n", node)
        # 抽取的信息存放在 metadata 字典的 excerpt_keywords 字段
        print("node.metadata: ", node.metadata)

    print("\n>>> Ingestion Pipeline:")
    embedding = get_embedding()
    pipeline = IngestionPipeline(
        name="CustomPipeline",
        project_name="CustomPipeline",  # 没啥用
        # 最重要的参数：配置一系列处理节点
        transformations=[
            sent_splitter,
            title_extractor,
            keyword_extractor,
            embedding
        ],
        # 可选参数
        # 可以直接配置 VectorStore 和 DocumentStore
        # vector_store=ChromaVectorStore(),
        # docstore=SimpleDocumentStore(),
    )
    pipeline_nodes = pipeline.run(documents=docs)
    print("len(nodes): ", len(pipeline_nodes))
    for node in pipeline_nodes:
        print("----------------")
        print("node:\n", node)
        print("node.metadata: ", node.metadata)
        print("node.embedding: ", node.embedding)

def read_docs() -> Sequence[BaseNode]:
    llm = get_llm()
    embedding = get_embedding()

    doc_dir = Path(r"C:\Users\Drivi\Documents\技术书籍\RAG-Documents")
    print("doc_dir: ", doc_dir, " exists: ", doc_dir.exists())
    files = list(doc_dir.glob("*.txt"))
    reader = SimpleDirectoryReader(input_files=files)
    docs: List[Document] = reader.load_data()

    sent_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    title_extractor = TitleExtractor(llm=llm)
    pipeline = IngestionPipeline(name="CustomPipeline", transformations=[sent_splitter, title_extractor, embedding])
    nodes = pipeline.run(documents=docs)
    return nodes


# %% ---------- Llama RAG-Store 使用 ----------
def rag_store_usage():
    """
    RAG-Store单独使用.
    Store配合Index使用时，通常要借助 StorageContext.
    """
    print("-------- Llama RAG-Store Usage --------")
    embedding = get_embedding()
    docs = read_docs()

    # 基于内存的向量存储
    # vec_store = SimpleVectorStore()
    # Chroma向量存储
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="test_collection")
    vec_store = ChromaVectorStore(chroma_collection=collection)

    # 主要API如下：
    # 插入
    vec_store.add(nodes=docs)

    # 查询
    query_vec = embedding.get_query_embedding(query="LlamaIndex RAG Pipeline")
    query = VectorStoreQuery(query_embedding=query_vec)
    result: VectorStoreQueryResult = vec_store.query(query=query)
    for idx in range(len(result.nodes)):
        print("----------------")
        print("id: ", result.ids[idx])
        print("similarity: ", result.similarities[idx])
        print("node:\n", result.nodes[idx])

    # 删除
    vec_store.delete(ref_doc_id=result.ids[0])
    vec_store.clear()


# %% ---------- Llama RAG-Index 使用 ----------
def rag_index_usage():
    """
    RAG-Index使用
    """
    print("-------- Llama RAG-Index Usage --------")
    embedding = get_embedding()
    docs = read_docs()

    # 1. 指定一个 BaseStore
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="test_collection")
    vec_store = ChromaVectorStore(chroma_collection=collection)

    # 2. 创建 Index，有多种方式
    # 2.1
    # 首先创建 StorageContext
    # storage_context = StorageContext.from_defaults(vector_store=vec_store)
    # 然后直接从 原始documents 创建，同时对docs进行处理
    # index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, transformations=[embedding])

    # 2.2 初始化一个 Index
    index = VectorStoreIndex.from_vector_store(vector_store=vec_store, embed_model=embedding)
    # 手动从 documents 构建Index
    index.build_index_from_nodes(nodes=docs)

    # 当然也可以手动使用 __init__ 方法创建 Index

    # 3. 查询，通过 as_retriever 方法 获取 Retriever 进行查询
    retriever: BaseRetriever = index.as_retriever()
    res = retriever.retrieve("LlamaIndex RAG Pipeline")
    print("query results:")
    for item in res:
        print("----------------")
        print("score: ", item.score)
        print("node:\n", item)

    # 插入
    index.insert(document=docs[0])
    # 删除
    index.delete(doc_id=docs[0].ref_doc_id)


# %% ---------- Llama RAG-Query 使用 ----------
def rag_query_usage():
    """
    RAG-Query使用.
    RAG-Query的主要步骤为：
    1. Retrieve
    2. PostProcessing
    3. ResponseSynthesis
    """
    print("-------- Llama RAG-Query Usage --------")
    llm = get_llm()
    embedding = get_embedding()
    docs = read_docs()

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="test_collection")
    vec_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vec_store, embed_model=embedding)
    index.build_index_from_nodes(nodes=docs)

    # ------ 高层API ------
    print(">>> Auto query_engine: ")
    # query_engine: BaseQueryEngine = index.as_query_engine()
    query_engine: RetrieverQueryEngine = index.as_query_engine(llm=llm, streaming=True)
    response: StreamingResponse = query_engine.query("Llama Index Pipeline usage")
    for chunk in response.response_gen:
        print(chunk, end="")
    print("\nresponse.metadata: ", response.metadata)
    # print("response.text: ", response.response_txt)

    # ------ 手动创建 QueryEngine，控制 PostProcessing 和 ResponseSynthesis ------
    print("\n>>> Manual query_engine: ")
    similarity = SimilarityPostprocessor(similarity_cutoff=0.2)
    response_synthesizer: BaseSynthesizer = get_response_synthesizer(
        llm=llm,
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        streaming=True
    )
    query_engine: RetrieverQueryEngine = index.as_query_engine(
        llm=llm,
        node_postprocessors=[similarity],
        response_synthesizer=response_synthesizer
    )
    response: StreamingResponse = query_engine.query("Llama Index Pipeline usage")
    for chunk in response.response_gen:
        print(chunk, end="")
    print("\nresponse.metadata: ", response.metadata)
    # print("response.text: ", response.response_txt)


# %% ---------- Llama RAG-Query-Router 使用 ----------
def rag_query_router_usage():
    """
    展示 RAG 过程中，涉及多个数据源（Retriever及其Index）时如何选择。
    本质上就是将多个 Retriever 封装成工具（RetrieverTool类），然后请求LLM选择使用哪个：
    - PydanticSelector: 使用LLM的 FunctionCalling API 来选择
    - LLMSelector: 使用LLM的 Completion API 来选择
    """
    print("-------- Llama RAG-Query-Router Usage --------")



# %% ---------- Llama-Agents/Workflow 使用 ----------
class SomeEvent(Event):
    hello: str

class SimpleWorkflow(Workflow):

    @step
    async def start_step(self, event: StartEvent) -> SomeEvent:
        hello: str = event.hello
        return SomeEvent(hello=hello)

    @step
    async def some_step(self, event: SomeEvent) -> StopEvent:
        print("some_step: ", event.hello)
        return StopEvent(result=f"{event.hello} -> Result")


async def llama_agent_usage():
    """
    展示 Llama-Agent 简单使用
    """
    print("-------- Llama-Agent Usage --------")
    workflow = SimpleWorkflow()
    hello = "Hello LlamaIndex"
    handler = workflow.run(hello=hello)
    result = await handler
    print(result)


def main():
    # llm_usage()
    # asyncio.run(llm_usage_async())
    # embedding_usage()
    # prompt_usage()
    # rag_loading_usage()
    # rag_store_usage()
    # rag_index_usage()
    # rag_query_usage()
    rag_query_router_usage()
    # asyncio.run(llama_agent_usage())


if __name__ == '__main__':
    main()
