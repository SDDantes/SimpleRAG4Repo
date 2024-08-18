import os
from typing import List, Dict, Any, Optional
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class RAGModel:
    """RAG模型集成类"""

    def __init__(
            self,
            retriever,
            llm: Optional[BaseLLM] = None,
            memory_key: str = "chat_history",
            return_source_documents: bool = True
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

        # 创建对话记忆
        # TODO：Deprecated，需替换为LangGraph实现
        self.memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=True,
            output_key='answer'
        )

        # 创建对话链
        self.qa_chain = self._create_qa_chain()

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
        return self.qa_chain.invoke({"question": question})

    def clear_history(self) -> None:
        """清除对话历史"""
        self.memory.clear()