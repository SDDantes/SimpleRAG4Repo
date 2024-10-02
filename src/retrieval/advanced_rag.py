from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document, BaseRetriever
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import logging
import ast

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class StructuredQuery(BaseModel):
    """用于解析的结构化查询"""
    rewritten_query: str = Field(description="重写后更清晰、更详细的查询")


class QuerySubquestions(BaseModel):
    """用于解析的查询子问题列表"""
    subquestions: List[str] = Field(description="原始查询的子问题列表")


class AdvancedRAGProcessor:
    """高级RAG处理器，提供查询重写、分解和自适应检索功能"""

    def __init__(self, llm: Optional[BaseLLM] = None, base_retriever: Optional[BaseRetriever] = None):
        """初始化高级RAG处理器"""
        # 如果未提供LLM，使用默认的OpenAI模型
        if llm is None:
            self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        else:
            self.llm = llm

        self.base_retriever = base_retriever

    def query_rewrite(self, original_query: str) -> str:
        """
        查询重写功能：将原始查询重写为更适合检索的形式
        """
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
        query_rewrite_chain = LLMChain(llm=self.llm, prompt=query_rewrite_template)

        # 执行查询重写
        result = query_rewrite_chain.run(original_query=original_query)

        try:
            # 解析结果
            parsed_result = parser.parse(result)
            return parsed_result.rewritten_query
        except Exception as e:
            # 解析失败时回退到原始查询
            print(f"查询重写解析错误: {str(e)}")
            print(f"原始LLM输出: {result}")
            return original_query

    def query_decomposition(self, original_query: str, max_subquestions: int = 3) -> List[str]:
        """
        查询分解功能：将复杂查询分解为多个简单子查询
        """
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
        query_decomp_chain = LLMChain(llm=self.llm, prompt=query_decomp_template)

        # 执行查询分解
        result = query_decomp_chain.run(original_query=original_query, max_subquestions=max_subquestions)

        try:
            # 解析结果
            parsed_result = parser.parse(result)
            return parsed_result.subquestions
        except Exception as e:
            # 解析失败时回退到原始查询
            print(f"查询分解解析错误: {str(e)}")
            print(f"原始LLM输出: {result}")
            return [original_query]

    def adaptive_retrieval(self, query: str, k: int = 5) -> Tuple[List[Document], Dict[str, Any]]:
        """
        自适应检索增强：根据查询特性选择最合适的检索策略
        返回: (检索到的文档列表, 检索元数据)
        """
        if self.base_retriever is None:
            raise ValueError("需要base_retriever来执行检索")

        # 分析查询复杂度的提示模板
        query_analysis_template = """分析以下用户查询的特性:

用户查询: {query}

请以JSON格式返回以下信息:
{{
    "complexity": "simple"|"medium"|"complex",
    "requires_code_examples": true|false,
    "is_technical": true|false,
    "recommended_strategy": "basic"|"decomposition"|"hybrid"
}}
"""

        # 创建LLM链分析查询
        analysis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=query_analysis_template,
                input_variables=["query"]
            )
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
        except json.JSONDecodeError as e:
            # 解析失败时使用默认策略
            print(f"查询分析解析错误。原始输出: {analysis_result}")
            logger.debug(e)
            analysis = {
                "complexity": "medium",
                "requires_code_examples": False,
                "is_technical": True,
                "recommended_strategy": "basic"
            }

        logger.debug(f"查询分析结果: {analysis}")

        # 根据分析结果选择检索策略
        retrieval_metadata = {"analysis": analysis, "strategy_used": "basic"}
        documents = []

        if analysis["recommended_strategy"] == "basic" or analysis["complexity"] == "simple":
            # 基本检索
            documents = self.base_retriever.get_relevant_documents(query, k=k)
            logger.debug(f"检索到文档{documents}")
            retrieval_metadata["strategy_used"] = "basic"

        elif analysis["recommended_strategy"] == "decomposition" or analysis["complexity"] == "complex":
            # 分解查询并检索
            subquestions = self.query_decomposition(query)
            all_docs = []

            for subq in subquestions:
                # 对每个子查询重写并检索
                rewritten_subq = self.query_rewrite(subq)
                docs = self.base_retriever.get_relevant_documents(rewritten_subq, k=max(2, k // len(subquestions)))
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
            retrieval_metadata["strategy_used"] = "decomposition"
            retrieval_metadata["subquestions"] = subquestions

        else:  # hybrid or fallback
            # 混合策略: 重写查询并检索
            rewritten_query = self.query_rewrite(query)
            documents = self.base_retriever.get_relevant_documents(rewritten_query, k=k)
            retrieval_metadata["strategy_used"] = "hybrid"
            retrieval_metadata["rewritten_query"] = rewritten_query

        return documents, retrieval_metadata