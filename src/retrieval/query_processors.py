import logging
from typing import List, Optional
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.utils.performance import timed, global_timing_stats

logger = logging.getLogger(__name__)


class StructuredQuery(BaseModel):
    """用于解析的结构化查询"""
    rewritten_query: str = Field(description="重写后更清晰、更详细的查询")


class QuerySubquestions(BaseModel):
    """用于解析的查询子问题列表"""
    subquestions: List[str] = Field(description="原始查询的子问题列表")


class QueryProcessors:
    """提供查询重写、分解等基础处理功能的工具类"""

    @staticmethod
    @timed("query_rewrite", stats_instance=global_timing_stats)
    def rewrite_query(original_query: str, llm: Optional[BaseLLM] = None) -> str:
        """
        查询重写功能：将原始查询重写为更适合检索的形式
        """
        # 如果未提供LLM，使用默认的OpenAI模型
        if llm is None:
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

        # 创建Pydantic解析器
        parser = PydanticOutputParser(pydantic_object=StructuredQuery)

        # 创建查询重写提示模板
        query_rewrite_template = PromptTemplate(
            template="""你是一个专业的查询优化器。你的任务是将用户的原始查询重写为更详细、更准确的查询，
以便更好地从公司的技术文档和代码仓库中检索相关信息。

原始查询: {original_query}

重写时考虑：
1. 增加技术术语和专业词汇
2. 更详细地阐述查询意图
3. 结构化表达，使查询更清晰
4. 保持查询的本质含义不变

{format_instructions}
""",
            input_variables=["original_query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # 创建LLM链
        query_rewrite_chain = LLMChain(llm=llm, prompt=query_rewrite_template)

        # 执行查询重写
        result = query_rewrite_chain.run(original_query=original_query)

        try:
            # 解析结果
            parsed_result = parser.parse(result)
            return parsed_result.rewritten_query
        except Exception as e:
            # 解析失败时回退到原始查询
            logger.warning(f"查询重写解析错误: {str(e)}")
            logger.debug(f"原始LLM输出: {result}")
            return original_query

    @staticmethod
    @timed("query_decomposition", stats_instance=global_timing_stats)
    def decompose_query(original_query: str, max_subquestions: int = 3, llm: Optional[BaseLLM] = None) -> List[str]:
        """
        查询分解功能：将复杂查询分解为多个简单子查询
        """
        # 如果未提供LLM，使用默认的OpenAI模型
        if llm is None:
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

        # 创建Pydantic解析器
        parser = PydanticOutputParser(pydantic_object=QuerySubquestions)

        # 创建查询分解提示模板
        query_decomp_template = PromptTemplate(
            template="""你是一个专业的查询分析器。你的任务是将复杂的用户查询分解为多个简单的子查询，
以便更全面地从公司的技术文档和代码仓库中检索相关信息。

原始查询: {original_query}

如果原始查询已经足够简单，可以返回一个只包含原始查询的列表。
否则，将其分解为不超过{max_subquestions}个子查询。

分解时考虑：
1. 每个子查询应关注原始查询的一个具体方面
2. 子查询应该相互补充，共同覆盖原始查询的全部含义
3. 每个子查询应该清晰、具体，便于检索

{format_instructions}
""",
            input_variables=["original_query", "max_subquestions"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # 创建LLM链
        query_decomp_chain = LLMChain(llm=llm, prompt=query_decomp_template)

        # 执行查询分解
        result = query_decomp_chain.run(original_query=original_query, max_subquestions=max_subquestions)

        try:
            # 解析结果
            parsed_result = parser.parse(result)
            return parsed_result.subquestions
        except Exception as e:
            # 解析失败时回退到原始查询
            logger.warning(f"查询分解解析错误: {str(e)}")
            logger.debug(f"原始LLM输出: {result}")
            return [original_query]

    @staticmethod
    @timed("generate_hypothetical_answer", stats_instance=global_timing_stats)
    def generate_hypothetical_answer(query: str, llm: Optional[BaseLLM] = None) -> str:
        """
        生成假设性答案用于HyDE检索
        """
        # 如果未提供LLM，使用默认的OpenAI模型
        if llm is None:
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

        # 创建生成假设性答案的提示模板
        hyde_prompt = PromptTemplate(
            template="""基于以下问题，生成一个假设性的、详细的回答。
这个回答应该像是从技术文档或代码库中摘录出来的。
不需要是完全正确的，但应该包含可能出现在相关文档中的专业术语和概念。

问题: {query}

假设性回答:""",
            input_variables=["query"]
        )

        # 创建生成假设性答案的链
        hyde_chain = LLMChain(llm=llm, prompt=hyde_prompt)

        # 生成假设性答案
        return hyde_chain.run(query=query)