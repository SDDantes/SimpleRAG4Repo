import os
from typing import List, Optional
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
    GitLoader
)
from langchain.schema import Document


class DocumentLoader:
    """加载各种文档类型的统一接口"""

    @staticmethod
    def load_markdown(file_path: str) -> List[Document]:
        """加载Markdown文档"""
        loader = UnstructuredMarkdownLoader(file_path)
        return loader.load()

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """加载纯文本文档"""
        loader = TextLoader(file_path)
        return loader.load()

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """加载PDF文档"""
        loader = PyPDFLoader(file_path)
        return loader.load()

    @staticmethod
    def load_html(file_path: str) -> List[Document]:
        """加载HTML文档"""
        loader = UnstructuredHTMLLoader(file_path)
        return loader.load()

    @staticmethod
    def load_git_repository(repo_path: str, branch: str = "main") -> List[Document]:
        """从Git仓库加载代码"""
        loader = GitLoader(
            repo_path=repo_path,
            branch=branch
        )
        return loader.load()

    @staticmethod
    def load_directory(directory_path: str, glob_pattern: Optional[str] = None) -> List[Document]:
        """递归加载目录中的所有文档，根据文件扩展名选择合适的加载器"""
        documents = []

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)

                # 根据文件扩展名选择加载器
                if file.endswith('.md'):
                    documents.extend(DocumentLoader.load_markdown(file_path))
                elif file.endswith('.txt'):
                    documents.extend(DocumentLoader.load_text(file_path))
                elif file.endswith('.pdf'):
                    documents.extend(DocumentLoader.load_pdf(file_path))
                elif file.endswith('.html') or file.endswith('.htm'):
                    documents.extend(DocumentLoader.load_html(file_path))
                # 忽略其他类型的文件

        return documents