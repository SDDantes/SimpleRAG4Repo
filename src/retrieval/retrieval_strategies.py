from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document, BaseRetriever
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI

from src.utils.performance import timed, global_timing_stats
from src.retrieval.query_processors import QueryProcessors


class RetrievalStrategy(ABC):
    """检索策略的抽象基类"""

    name: str = "base_strategy"
    description: str = "基础检索策略"

    def __init__(self, retriever: BaseRetriever, llm: Optional[BaseLLM] = None):
        self.retriever = retriever
        if llm is None:
            self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        else:
            self.llm = llm

    @abstractmethod
    @timed(stats_instance=global_timing_stats)
    def retrieve(self, query: str, k: int = 5, **kwargs) -> Tuple[List[Document], Dict[str, Any]]:
        """
        执行检索操作
        返回: (检索到的文档, 检索元数据)
        """
        pass

    @classmethod
    def get_strategy_info(cls) -> Dict[str, str]:
        """获取策略信息"""
        return {
            "name": cls.name,
            "description": cls.description
        }


class BasicRetrievalStrategy(RetrievalStrategy):
    """基本检索策略，直接使用检索器"""

    name = "basic"
    description = "直接使用基础检索器进行检索，适用于简单查询"

    @timed("basic_retrieval", stats_instance=global_timing_stats)
    def retrieve(self, query: str, k: int = 5, **kwargs) -> Tuple[List[Document], Dict[str, Any]]:
        """直接使用基础检索器检索文档"""
        documents = self.retriever.get_relevant_documents(query, k=k)
        metadata = {
            "strategy": self.name,
            "k": k,
            "query": query
        }
        return documents, metadata


class QueryRewriteRetrievalStrategy(RetrievalStrategy):
    """查询重写检索策略"""

    name = "query_rewrite"
    description = "重写原始查询为更适合检索的形式，然后执行检索"

    @timed("query_rewrite_retrieval", stats_instance=global_timing_stats)
    def retrieve(self, query: str, k: int = 5, **kwargs) -> Tuple[List[Document], Dict[str, Any]]:
        """重写查询并检索"""
        # 使用查询处理器进行查询重写
        rewritten_query = QueryProcessors.rewrite_query(query, llm=self.llm)

        # 使用重写后的查询检索文档
        documents = self.retriever.get_relevant_documents(rewritten_query, k=k)

        metadata = {
            "strategy": self.name,
            "k": k,
            "original_query": query,
            "rewritten_query": rewritten_query
        }

        return documents, metadata


class QueryDecompositionStrategy(RetrievalStrategy):
    """查询分解检索策略"""

    name = "decomposition"
    description = "将复杂查询分解为多个简单子查询，分别检索并合并结果"

    @timed("query_decomposition_retrieval", stats_instance=global_timing_stats)
    def retrieve(self, query: str, k: int = 5, **kwargs) -> Tuple[List[Document], Dict[str, Any]]:
        """分解查询并检索"""
        # 使用查询处理器分解查询
        subquestions = QueryProcessors.decompose_query(query, llm=self.llm)

        all_docs = []

        # 为每个子查询检索文档
        for subq in subquestions:
            # 对每个子查询重写并检索
            rewritten_subq = QueryProcessors.rewrite_query(subq, llm=self.llm)
            docs = self.retriever.get_relevant_documents(
                rewritten_subq,
                k=max(2, k // len(subquestions))
            )
            all_docs.extend(docs)

        # 去重
        seen_contents = set()
        documents = []
        for doc in all_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                documents.append(doc)

        # 如果需要，限制文档数量
        documents = documents[:k]

        metadata = {
            "strategy": self.name,
            "k": k,
            "original_query": query,
            "subquestions": subquestions
        }

        return documents, metadata


class HyDERetrievalStrategy(RetrievalStrategy):
    """假设文档检索(HyDE)策略"""

    name = "hyde"
    description = "生成假设性答案，然后使用该答案作为检索查询"

    @timed("hyde_retrieval", stats_instance=global_timing_stats)
    def retrieve(self, query: str, k: int = 5, **kwargs) -> Tuple[List[Document], Dict[str, Any]]:
        """使用HyDE技术检索文档"""
        # 使用查询处理器生成假设性答案
        with global_timing_stats.measure("hyde_generate_answer"):
            hypothetical_answer = QueryProcessors.generate_hypothetical_answer(query, llm=self.llm)

        # 使用假设性答案作为检索查询
        with global_timing_stats.measure("hyde_retrieve_with_answer"):
            documents = self.retriever.get_relevant_documents(hypothetical_answer, k=k)

        metadata = {
            "strategy": self.name,
            "k": k,
            "original_query": query,
            "hypothetical_answer": hypothetical_answer[:300] + "..." if len(
                hypothetical_answer) > 300 else hypothetical_answer
        }

        return documents, metadata


# 策略注册表
class StrategyRegistry:
    """策略注册表，用于管理所有可用的检索策略"""

    _strategies = {}

    @classmethod
    def register(cls, strategy_class):
        """注册新策略"""
        cls._strategies[strategy_class.name] = strategy_class
        return strategy_class

    @classmethod
    def get_strategy(cls, name: str, retriever: BaseRetriever, llm: Optional[BaseLLM] = None):
        """获取指定名称的策略实例"""
        if name not in cls._strategies:
            raise ValueError(f"未知策略: {name}")
        return cls._strategies[name](retriever=retriever, llm=llm)

    @classmethod
    def list_strategies(cls):
        """列出所有可用策略"""
        return [strategy.get_strategy_info() for strategy in cls._strategies.values()]

    @classmethod
    def get_strategy_names(cls):
        """获取所有策略名称"""
        return list(cls._strategies.keys())


# 注册所有策略
StrategyRegistry.register(BasicRetrievalStrategy)
StrategyRegistry.register(QueryRewriteRetrievalStrategy)
StrategyRegistry.register(QueryDecompositionStrategy)
StrategyRegistry.register(HyDERetrievalStrategy)