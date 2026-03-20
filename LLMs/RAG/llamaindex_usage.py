"""
LlamaIndex使用研究
"""
# %% ---------- LlamaIndex 核心包 ----------
# --- LlamaIndex 抽象基础 ---
from llama_index.core import Settings
# from llama_index.core.schema import BaseComponent, BaseNode
from llama_index.core.schema import NodeRelationship, Node, TextNode, ImageNode, IndexNode, Document
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.base.response.schema import Response, PydanticResponse
# --- LlamaIndex 具体组件 ---
from llama_index.core.llms import (
    LLMMetadata, MockLLM, LLM, CustomLLM, MessageRole, ChatMessage, ChatResponse, CompletionResponse,
    TextBlock, DocumentBlock
)
from llama_index.core.prompts import Prompt, PromptTemplate, ChatPromptTemplate, RichPromptTemplate
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import (
    NodeParser, SimpleNodeParser, TokenTextSplitter, TextSplitter, SentenceSplitter, MarkdownNodeParser,
    HTMLNodeParser, CodeSplitter, SentenceWindowNodeParser, HierarchicalNodeParser
)
from llama_index.core.extractors import BaseExtractor, TitleExtractor, KeywordExtractor, SummaryExtractor, QuestionsAnsweredExtractor, DocumentContextExtractor
from llama_index.core.ingestion import IngestionCache, IngestionPipeline, DocstoreStrategy, run_transformations, arun_transformations
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices import (
    VectorStoreIndex, SummaryIndex, DocumentSummaryIndex, KeywordTableIndex, SimpleKeywordTableIndex, TreeIndex,
    KnowledgeGraphIndex, PandasIndex,
)
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery, VectorStoreInfo
# from llama_index.core import StorageContext
from llama_index.core.storage import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore, DocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.core.storage.kvstore import SimpleKVStore, RedisKVStore, FirestoreKVStore
from llama_index.core.graph_stores import SimpleGraphStore, PropertyGraphStore, SimplePropertyGraphStore, EntityNode, LabelledNode, ChunkNode, Relation
from llama_index.core.retrievers import (
    VectorIndexRetriever, VectorIndexAutoRetriever, SummaryIndexRetriever, KGTableRetriever, KnowledgeGraphRAGRetriever,
    BM25Retriever
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
