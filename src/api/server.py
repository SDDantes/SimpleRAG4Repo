import os
import traceback
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 导入自定义模块
from src.vectordb.store import VectorStore
from src.retrieval.retriever import EnhancedRetriever
from src.llm.models import RAGModel

# 初始化全局变量
vector_store = None
rag_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global vector_store, rag_model

    try:
        # 启动时初始化组件
        logger.info("初始化服务组件...")
        # 获取向量数据库路径
        vector_db_path = os.getenv("VECTORDB_PATH", "./vectorstore")
        logger.debug(f"向量数据库路径: {vector_db_path}")

        # 初始化向量存储
        logger.info("初始化向量存储...")
        vector_store = VectorStore(persist_directory=vector_db_path)

        # 获取基础检索器
        logger.info("创建基础检索器...")
        base_retriever = vector_store.get_retriever(search_type="mmr", k=5)

        # 创建增强检索器
        logger.info("创建增强检索器...")
        enhanced_retriever = EnhancedRetriever(base_retriever=base_retriever)

        # 创建RAG模型
        logger.info("创建RAG模型...")
        rag_model = RAGModel(retriever=enhanced_retriever.compression_retriever)
        logger.info("所有组件初始化完成")
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    yield

    # 清理资源
    logger.info("正在关闭服务并清理资源...")


# 初始化FastAPI应用
app = FastAPI(title="公司规范与代码知识库助手", lifespan=lifespan)


# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"全局异常: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}", "traceback": traceback.format_exc()},
    )


class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str
    clear_history: bool = False


class ResponseItem(BaseModel):
    """响应项模型"""
    answer: str
    # PyCharm错误检查
    # noinspection PyDataclass
    source_documents: List[Dict[str, Any]] = Field(default_factory=list)


@app.post("/query", response_model=ResponseItem)
async def query(request: QueryRequest):
    """处理查询请求"""
    logger.info(f"收到查询请求: {request.question}")

    if rag_model is None:
        logger.error("RAG模型未初始化")
        raise HTTPException(status_code=500, detail="RAG模型未初始化")

    # 如果请求要求清除历史，则清除
    if request.clear_history:
        logger.info("清除对话历史")
        rag_model.clear_history()

    # 检查question是否为空字符串
    if not request.question.strip():
        logger.warning("查询问题为空")
        return {
            "answer": "查询问题不能为空",
            "source_documents": []
        }

    # 执行查询
    try:
        logger.debug("开始执行RAG查询")
        result = rag_model.query(request.question)
        logger.debug(f"查询结果: {result.get('answer', '')[:100]}...")

        # 格式化源文档
        source_docs = []
        if result.get("source_documents"):
            logger.debug(f"查询返回了{len(result['source_documents'])}个源文档")
            for doc in result["source_documents"]:
                source_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })

        return {
            "answer": result["answer"],
            "source_documents": source_docs
        }
    except Exception as e:
        error_msg = f"查询处理错误: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/health")
async def health_check():
    """健康检查端点"""
    logger.debug("收到健康检查请求")
    return {"status": "healthy"}