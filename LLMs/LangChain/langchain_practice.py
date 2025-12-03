"""
练习 LangChain-Core / LangChain v1.0 的使用
"""
from typing import Annotated, List, TypedDict, Dict, Union
# from typing_extensions import TypedDict
# ----------------- LangChain-Core -----------------
from langchain_core.runnables import Runnable, RunnableConfig, RunnableBinding, RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import BaseMessage, BaseMessageChunk, ContentBlock
# 下面的 Message 类，就是 langchain.messages 里实际导入的对象
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, ChatMessage
from langchain_core.prompts import StringPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder, ChatMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, FewShotChatMessagePromptTemplate, PipelinePromptTemplate
# ----------------- LangChain -----------------
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.embeddings import Embeddings, init_embeddings
from langchain.tools import BaseTool, tool, InjectedToolArg, InjectedState, ToolException
# ------ v1.0 里统一的 agent 创建API ------
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
# ------ middleware，v1.0版本一个更新亮点 ------
from langchain.agents.middleware import (
    AgentMiddleware, AgentState, ModelRequest, ModelResponse,
    before_agent, after_agent, before_model, after_model, wrap_model_call, wrap_tool_call, hook_config
)
# 自带的 middleware 实现
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware, ModelCallLimitMiddleware, \
    ToolCallLimitMiddleware
# ----------------- LangGraph组件 -----------------
from langgraph.runtime import Runtime
# ----------------- LLM模型提供商 -----------------
# 一线模型厂商（比如OpenAI），建议直接从第三方包里导入
from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama


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





if __name__ == '__main__':
    ...
