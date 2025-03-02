from typing import List, Union
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    Language,
    PythonCodeTextSplitter
)


class DocumentProcessor:
    """处理和分割文档的统一接口"""

    @staticmethod
    def split_text(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """分割普通文本文档"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)

    @staticmethod
    def split_markdown(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """分割Markdown文档"""
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return markdown_splitter.split_documents(documents)

    @staticmethod
    def split_code(documents: List[Document], language: str = "python", chunk_size: int = 1000,
                   chunk_overlap: int = 200) -> List[Document]:
        """分割代码文档，保持代码结构完整性"""
        if language.lower() == "python":
            code_splitter = PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            # 对于其他语言使用通用代码分割器
            code_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language(language),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        return code_splitter.split_documents(documents)

    @staticmethod
    def process_documents(documents: List[Document], document_type: str = "text",
                          chunk_size: int = 1000, chunk_overlap: int = 200,
                          language: str = "python") -> List[Document]:
        """根据文档类型处理文档"""
        if document_type == "markdown":
            return DocumentProcessor.split_markdown(documents, chunk_size, chunk_overlap)
        elif document_type == "code":
            return DocumentProcessor.split_code(documents, language, chunk_size, chunk_overlap)
        else:  # 默认为文本
            return DocumentProcessor.split_text(documents, chunk_size, chunk_overlap)

    @staticmethod
    def add_source_metadata(documents: List[Document], source: str) -> List[Document]:
        """为文档添加源信息元数据"""
        for doc in documents:
            doc.metadata["source"] = source
        return documents