import os
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma


class VectorStore:
    """向量数据库管理类"""

    def __init__(
            self,
            persist_directory: str,
            embedding_model: Optional[Embeddings] = None,
            collection_name: str = "company_docs"
    ):
        """初始化向量存储"""
        self.persist_directory = persist_directory

        # 如果没有提供嵌入模型，使用OpenAI嵌入
        if embedding_model is None:
            self.embedding_model = OpenAIEmbeddings()
        else:
            self.embedding_model = embedding_model

        self.collection_name = collection_name

        # 确保存储目录存在
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化或加载向量存储
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model,
            collection_name=collection_name
        )

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到向量存储"""
        self.db.add_documents(documents)
        # Since Chroma 0.4.x the manual persistence method is no longer supported
        # as docs are automatically persisted.
        #self.db.persist()

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """执行相似度搜索"""
        return self.db.similarity_search(query, k=k, filter=filter)

    def mmr_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """执行最大边际相关性搜索（提高多样性）"""
        return self.db.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )

    def get_retriever(self, search_type: str = "mmr", **kwargs):
        """获取检索器"""
        return self.db.as_retriever(search_type=search_type, search_kwargs=kwargs)

    @classmethod
    def from_documents(
            cls,
            documents: List[Document],
            persist_directory: str,
            embedding_model: Optional[Embeddings] = None,
            collection_name: str = "company_docs"
    ):
        """从文档创建向量存储"""
        instance = cls(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            collection_name=collection_name
        )
        instance.add_documents(documents)
        return instance

    @classmethod
    def get_huggingface_embeddings(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """获取HuggingFace嵌入模型"""
        return HuggingFaceEmbeddings(model_name=model_name)