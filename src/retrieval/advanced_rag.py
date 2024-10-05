from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document, BaseRetriever
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import json
import logging
import ast

from src.utils.performance import timed, global_timing_stats
# 导入策略注册表，不再需要导入具体的查询处理功能
from src.retrieval.retrieval_strategies import StrategyRegistry
# 导入查询处理器，用于自己需要的查询处理
from src.retrieval.query_processors import QueryProcessors

logger = logging.getLogger(__name__)


class QueryAnalysis(BaseModel):
    """查询分析结果"""
    complexity: str = Field(description="查询复杂度: simple, medium, complex")
    requires_code_examples: bool = Field(description="查询是否需要代码示例")
    is_technical: bool = Field(description="是否是技术性查询")
    recommended_strategy: str = Field(description="推荐的检索策略")
    explanation: str = Field(description="策略选择的解释")


class AdvancedRAGProcessor:
    """高级RAG处理器，提供查询分析和自适应检索功能"""

    def __init__(self, llm: Optional[BaseLLM] = None, base_retriever: Optional[BaseRetriever] = None):
        """初始化高级RAG处理器"""
        # 如果未提供LLM，使用默认的OpenAI模型
        if llm is None:
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        else:
            self.llm = llm

        self.base_retriever = base_retriever

    # 这些方法现在是从QueryProcessors调用的，保留这些公共方法作为兼容接口
    def query_rewrite(self, original_query: str) -> str:
        """查询重写功能（保留兼容性）"""
        return QueryProcessors.rewrite_query(original_query, llm=self.llm)

    def query_decomposition(self, original_query: str, max_subquestions: int = 3) -> List[str]:
        """查询分解功能（保留兼容性）"""
        return QueryProcessors.decompose_query(original_query, max_subquestions, llm=self.llm)

    @timed("analyze_query", stats_instance=global_timing_stats)
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析查询特性，推荐最佳检索策略
        """
        # 获取可用策略列表
        available_strategies = StrategyRegistry.get_strategy_names()
        strategies_str = ", ".join(available_strategies)

        # 创建查询分析提示模板
        query_analysis_template = PromptTemplate(
            template="""分析以下用户查询的特性，并推荐最佳检索策略:

用户查询: {query}

可用的检索策略:
{strategies}

请以JSON格式返回以下信息:
{{
    "complexity": "simple"|"medium"|"complex",
    "requires_code_examples": true|false,
    "is_technical": true|false,
    "recommended_strategy": "{strategy_options}",
    "explanation": "策略选择的简短解释"
}}

策略选择指南:
- basic: 适用于简单、直接的查询
- query_rewrite: 适用于需要澄清、扩展的查询
- decomposition: 适用于复杂的、多方面、宽泛的或者多步骤查询
- hyde: 适用于需要详细技术解释或代码示例的查询
""",
            input_variables=["query"],
            partial_variables={
                "strategies": strategies_str,
                "strategy_options": "|".join(available_strategies)
            }
        )

        # 创建LLM链分析查询
        analysis_chain = LLMChain(
            llm=self.llm,
            prompt=query_analysis_template
        )

        # 执行查询分析
        analysis_result = analysis_chain.run(query=query)

        try:
            # 解析分析结果
            if analysis_result.startswith("```json"):
                # 如果是JSON代码块，提取并解析JSON
                analysis = json.loads(analysis_result[7:-3])
            elif isinstance(analysis_result, str):
                # 如果是字符串但不是JSON代码块，尝试用ast解析
                analysis = ast.literal_eval(analysis_result)
            else:
                # 如果已经是字典对象，直接使用
                analysis = analysis_result

            # 验证推荐的策略是否可用
            if analysis["recommended_strategy"] not in available_strategies:
                logger.warning(f"推荐的策略 {analysis['recommended_strategy']} 不可用，回退到basic")
                analysis["recommended_strategy"] = "basic"
                analysis["explanation"] += " (原推荐策略不可用，回退到basic)"

            return analysis
        except (json.JSONDecodeError, SyntaxError, ValueError):
            # 解析失败时使用默认策略
            logger.warning(f"查询分析解析错误。原始输出: {analysis_result}")
            return {
                "complexity": "medium",
                "requires_code_examples": False,
                "is_technical": True,
                "recommended_strategy": "basic",
                "explanation": "解析失败，使用默认策略"
            }

    @timed("adaptive_retrieval", stats_instance=global_timing_stats)
    def adaptive_retrieval(self, query: str, k: int = 5) -> Tuple[List[Document], Dict[str, Any]]:
        """
        自适应检索增强：根据查询特性选择最合适的检索策略
        返回: (检索到的文档列表, 检索元数据)
        """
        if self.base_retriever is None:
            raise ValueError("需要base_retriever来执行检索")

        # 分析查询，选择最佳策略
        analysis = self.analyze_query(query)

        # 获取推荐的策略
        strategy_name = analysis["recommended_strategy"]

        # 创建策略实例
        try:
            strategy = StrategyRegistry.get_strategy(
                name=strategy_name,
                retriever=self.base_retriever,
                llm=self.llm
            )
        except ValueError:
            logger.warning(f"策略 {strategy_name} 不可用，回退到basic")
            strategy = StrategyRegistry.get_strategy(
                name="basic",
                retriever=self.base_retriever,
                llm=self.llm
            )
            analysis["recommended_strategy"] = "basic"
            analysis["explanation"] += " (原推荐策略不可用，回退到basic)"

        # 使用选定的策略执行检索
        documents, strategy_metadata = strategy.retrieve(query, k=k)

        # 合并所有元数据
        retrieval_metadata = {
            "analysis": analysis,
            "strategy_used": strategy_name,
            **strategy_metadata
        }

        # 记录性能统计
        retrieval_metadata["performance"] = global_timing_stats.get_summary()

        return documents, retrieval_metadata