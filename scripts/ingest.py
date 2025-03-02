import os
import argparse
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document

# 导入自定义模块
from src.data_ingestion.loaders import DocumentLoader
from src.data_ingestion.processors import DocumentProcessor
from src.vectordb.store import VectorStore

# 加载环境变量
load_dotenv()


def process_documents(
        input_dir: str,
        document_type: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        language: str = "python"
) -> List[Document]:
    """处理指定目录下的文档"""
    print(f"正在从 {input_dir} 加载 {document_type} 类型的文档...")

    # 加载文档
    documents = DocumentLoader.load_directory(input_dir)

    print(f"加载了 {len(documents)} 个文档，正在处理...")

    # 处理文档
    processed_docs = DocumentProcessor.process_documents(
        documents=documents,
        document_type=document_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        language=language
    )

    # 添加源信息
    processed_docs = DocumentProcessor.add_source_metadata(
        documents=processed_docs,
        source=input_dir
    )

    print(f"处理完成，共 {len(processed_docs)} 个文档块")
    return processed_docs


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据摄取工具")
    parser.add_argument("--docs_dir", type=str, help="文档目录路径")
    parser.add_argument("--code_dir", type=str, help="代码目录路径")
    parser.add_argument("--doc_type", type=str, default="markdown", help="文档类型 (markdown, text)")
    parser.add_argument("--code_language", type=str, default="python", help="代码语言")
    parser.add_argument("--chunk_size", type=int, default=1000, help="文档块大小")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="文档块重叠大小")
    parser.add_argument("--vector_db_path", type=str, default=os.getenv("VECTORDB_PATH", "./vectorstore"),
                        help="向量数据库路径")

    args = parser.parse_args()

    # 初始化向量存储
    vector_store = VectorStore(
        persist_directory=args.vector_db_path
    )

    all_docs = []

    # 处理文档
    if args.docs_dir:
        docs = process_documents(
            input_dir=args.docs_dir,
            document_type=args.doc_type,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        all_docs.extend(docs)

    # 处理代码
    if args.code_dir:
        code_docs = process_documents(
            input_dir=args.code_dir,
            document_type="code",
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            language=args.code_language
        )
        all_docs.extend(code_docs)

    # 如果没有文档，退出
    if not all_docs:
        print("没有文档可处理，请指定 --docs_dir 或 --code_dir")
        return

    # 添加到向量存储
    print(f"正在将 {len(all_docs)} 个文档块添加到向量数据库...")
    vector_store.add_documents(all_docs)
    print("完成！文档已成功添加到向量数据库")


if __name__ == "__main__":
    main()