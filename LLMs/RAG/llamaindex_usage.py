"""
LlamaIndex使用研究
"""
import asyncio
# %% ---------- LlamaIndex 核心包 ----------
# --- LlamaIndex 抽象基础 ---
from llama_index.core import Settings
# from llama_index.core.schema import BaseComponent, BaseNode
from llama_index.core.schema import NodeRelationship, Node, TextNode, ImageNode, IndexNode, Document
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.base.response.schema import Response, PydanticResponse
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
from llama_index.core.ingestion import IngestionCache, IngestionPipeline, DocstoreStrategy, run_transformations, arun_transformations
# --- RAG: Indexing 组件 ---
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices import (
    VectorStoreIndex, SummaryIndex, DocumentSummaryIndex, KeywordTableIndex, SimpleKeywordTableIndex, TreeIndex,
    KnowledgeGraphIndex, PandasIndex,
)
# --- RAG: Storing 组件 ---
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery, VectorStoreInfo
# from llama_index.core import StorageContext
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore, DocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
# from llama_index.core.storage.kvstore import SimpleKVStore, RedisKVStore, FirestoreKVStore
from llama_index.core.graph_stores import (
    SimpleGraphStore, PropertyGraphStore, SimplePropertyGraphStore, EntityNode, LabelledNode, ChunkNode, Relation
)
# --- RAG: Quering 组件 ---
from llama_index.core.extractors import (
    BaseExtractor, TitleExtractor, KeywordExtractor, SummaryExtractor,
    QuestionsAnsweredExtractor, DocumentContextExtractor
)
from llama_index.core.retrievers import (
    VectorIndexRetriever, VectorIndexAutoRetriever, SummaryIndexRetriever,
    KGTableRetriever, KnowledgeGraphRAGRetriever,
    # BM25Retriever
)
from llama_index.core.selectors import (
    BaseSelector, SelectorResult, SingleSelection, MultiSelection,
    LLMSingleSelector, LLMMultiSelector, EmbeddingSingleSelector, PydanticSingleSelector, PydanticMultiSelector
)
from llama_index.core.postprocessor import (
    SimilarityPostprocessor, KeywordNodePostprocessor, DocumentWithRelevance, StructuredLLMRerank, LLMRerank
)
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    ResponseMode, BaseSynthesizer, Refine, SimpleSummarize, TreeSummarize, CompactAndRefine, Accumulate
)
# --- RAG: Evaluating 组件 ---
# --- RAG: 可观测性 组件 ---
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
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.embeddings
from llama_index.readers.file import (
    CSVReader, DocxReader, PDFReader, MarkdownReader, PyMuPDFReader, UnstructuredReader
)
from llama_index.vector_stores.chroma import ChromaVectorStore

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
    prompt_usage()
    # asyncio.run(llama_agent_usage())


if __name__ == '__main__':
    main()
