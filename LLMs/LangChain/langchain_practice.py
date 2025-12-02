"""
练习 LangChain-Core / LangChain v1.0 的使用
"""
# ----------------- LangChain-Core -----------------
from langchain_core.runnables import Runnable, RunnableConfig, RunnableBinding, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, ChatMessage
from langchain_core.prompts import StringPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder, ChatMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, FewShotChatMessagePromptTemplate, PipelinePromptTemplate
# ----------------- LangChain -----------------
from langchain.messages import ContentBlock, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.embeddings import Embeddings, init_embeddings
from langchain.tools import BaseTool, InjectedToolArg, InjectedState, tool, ToolException
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
