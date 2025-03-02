from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever


class EnhancedRetriever:
    """增强检索器，包含各种先进检索技术"""

    def __init__(self, base_retriever: BaseRetriever, llm: Optional[BaseLLM] = None):
        """初始化增强检索器"""
        self.base_retriever = base_retriever

        # 如果没有提供LLM，使用默认的ChatOpenAI
        if llm is None:
            self.llm = ChatOpenAI(temperature=0)
        else:
            self.llm = llm

        # 初始化增强检索器
        self._init_enhanced_retrievers()

    def _init_enhanced_retrievers(self):
        """初始化各种增强检索器"""
        # 上下文压缩检索器（提取最相关内容）
        self.compression_retriever = self._create_compression_retriever()

        # 多查询检索器（生成多个查询变体）
        self.multi_query_retriever = self._create_multi_query_retriever()

    def _create_compression_retriever(self) -> ContextualCompressionRetriever:
        """创建上下文压缩检索器"""
        compressor = LLMChainExtractor.from_llm(self.llm)
        return ContextualCompressionRetriever(
            base_retriever=self.base_retriever,
            base_compressor=compressor
        )

    def _create_multi_query_retriever(self) -> MultiQueryRetriever:
        """创建多查询检索器"""
        return MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )

    def retrieve(self, query: str, retriever_type: str = "compression", **kwargs) -> List[Document]:
        """根据指定的检索器类型执行检索"""
        if retriever_type == "base":
            return self.base_retriever.get_relevant_documents(query, **kwargs)
        elif retriever_type == "compression":
            return self.compression_retriever.get_relevant_documents(query, **kwargs)
        elif retriever_type == "multi_query":
            return self.multi_query_retriever.get_relevant_documents(query, **kwargs)
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")