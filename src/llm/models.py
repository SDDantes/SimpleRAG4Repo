import os
from typing import Dict, Any, Optional
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

from src.retrieval.advanced_rag import AdvancedRAGProcessor

# 加载环境变量
load_dotenv()

class RAGModel:
    """RAG模型集成类"""

    def __init__(
            self,
            retriever,
            llm: Optional[BaseLLM] = None,
            memory_key: str = "chat_history",
            return_source_documents: bool = True,
            use_advanced_rag: bool = True
    ):
        """初始化RAG模型"""
        # 如果没有提供LLM，使用默认的ChatOpenAI
        if llm is None:
            self.llm = ChatOpenAI(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=os.getenv("OPENAI_BASE_URL"),
                model_name="gpt-4o-mini",
                temperature=0.2)
        else:
            self.llm = llm

        self.retriever = retriever
        self.memory_key = memory_key
        self.return_source_documents = return_source_documents
        self.use_advanced_rag = use_advanced_rag

        # 初始化高级RAG处理器
        if use_advanced_rag:
            self.advanced_rag = AdvancedRAGProcessor(
                llm=self.llm,
                base_retriever=self.retriever
            )

        # 创建对话记忆
        # TODO：Deprecated，需替换为LangGraph实现
        self.memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=True,
            output_key='answer'
        )

        # 创建对话链
        self.qa_chain = self._create_qa_chain()

        # 存储最近一次的查询元数据
        self.last_retrieval_metadata = {}

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """创建问答链"""
        # 创建自定义提示模板以更好地处理代码和技术文档
        condense_question_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name=self.memory_key),
            ("human", "{question}")
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位公司内部的技术助手，帮助员工理解公司的代码规范和技术文档。
回答要精确、专业，并基于检索到的上下文信息。如果上下文中没有足够信息，请清晰说明。
对于代码相关问题，尽量给出具体的代码示例。"""),
            ("human", """基于以下上下文信息回答问题:
{context}

问题: {question}""")
        ])

        # 此链分为两步：
        # 1. 将输入问题与对话历史压缩为单个问题
        # 2. 使用压缩后的问题和检索到的文档作为上下文，回答问题
        # 还可以使用condense_question_llm来指定压缩用LLM
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=self.return_source_documents,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            get_chat_history=lambda h: h,
            chain_type="stuff"  # 明确指定链类型
        )

    def query(self, question: str) -> Dict[str, Any]:
        """查询RAG模型"""
        if not self.use_advanced_rag:
            # 使用标准RAG流程
            return self.qa_chain.invoke({"question": question})

        # 使用高级RAG流程
        # 步骤1: 将历史对话和当前问题结合，生成独立查询
        if len(self.memory.chat_memory.messages) > 0:
            # 使用记忆中的历史和当前问题生成独立查询
            history = self.memory.load_memory_variables({})
            standalone_question = self._get_standalone_question(
                history[self.memory_key],
                question
            )
        else:
            standalone_question = question

        # 步骤2: 使用自适应检索获取相关文档
        docs, retrieval_metadata = self.advanced_rag.adaptive_retrieval(standalone_question)
        self.last_retrieval_metadata = retrieval_metadata

        # 步骤3: 使用检索到的文档回答问题
        result = self.qa_chain.combine_docs_chain.invoke({
            "input_documents": docs,
            "question": question
        })

        # 构造返回结果
        response = {
            "question": question,
            "answer": result["output_text"],
        }

        if self.return_source_documents:
            response["source_documents"] = docs

        # 更新对话历史
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(result["output_text"])

        # 添加高级RAG元数据
        response["retrieval_metadata"] = retrieval_metadata

        return response

    def _get_standalone_question(self, chat_history, question) -> str:
        """将对话历史和当前问题结合，生成独立查询"""
        # 如果记忆刚刚初始化，直接返回问题
        if not chat_history:
            return question

        # 创建一个提示来生成独立问题
        prompt = PromptTemplate(
            template="""给定以下对话历史和最新问题，请生成一个独立的问题，包含解答最新问题所需的所有上下文信息。

对话历史:
{chat_history}

最新问题: {question}

独立问题:""",
            input_variables=["chat_history", "question"]
        )

        standalone_chain = LLMChain(llm=self.llm, prompt=prompt)
        return standalone_chain.run(chat_history=chat_history, question=question)

    def clear_history(self) -> None:
        """清除对话历史"""
        self.memory.clear()
        self.last_retrieval_metadata = {}