import os
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple

# 加载环境变量
load_dotenv()

# 设置页面
st.set_page_config(
    page_title="公司规范与代码知识库助手",
    page_icon="💻",
    layout="wide"
)

# API端点
API_URL = os.getenv("API_URL", "http://localhost:8000")


def query_api(question, clear_history=False, advanced_rag=True):
    """调用API进行查询"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "question": question,
                "clear_history": clear_history,
                "advanced_rag": advanced_rag
            }
        )
        return response.json()
    except Exception as e:
        st.error(f"API调用错误: {str(e)}")
        return None


# 将展开器内容生成抽象为单独的函数
def generate_expander_content(response: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    根据API响应生成扩展器内容

    Args:
        response: API响应字典

    Returns:
        包含(标题, 内容)元组的列表
    """
    expanders_content = []

    # 如果有源文档，准备显示内容
    if "source_documents" in response and response["source_documents"]:
        source_docs_content = []
        for i, doc in enumerate(response["source_documents"]):
            source_docs_content.append(f"**源 {i + 1}**: {doc.get('metadata', {}).get('source', '未知')}")
            source_docs_content.append(f"```\n{doc.get('page_content', '无内容')}\n```")
            source_docs_content.append("---")

        expanders_content.append(("查看源文档", "\n".join(source_docs_content)))

    # 如果有检索元数据，准备显示内容
    if "retrieval_metadata" in response and response["retrieval_metadata"]:
        metadata = response["retrieval_metadata"]
        metadata_content = []

        # 检索策略
        if "strategy_used" in metadata:
            strategy_map = {
                "basic": "基本检索",
                "query_rewrite": "查询重写",
                "decomposition": "查询分解",
                "hyde": "假设文档检索",
                "hybrid": "混合策略"
            }
            strategy = strategy_map.get(metadata["strategy_used"], metadata["strategy_used"])
            metadata_content.append(f"**检索策略**: {strategy}")

        # 查询分析
        if "analysis" in metadata:
            analysis = metadata["analysis"]
            metadata_content.append("**查询分析**:")
            complexity_map = {
                "simple": "简单",
                "medium": "中等",
                "complex": "复杂"
            }
            complexity = complexity_map.get(analysis.get("complexity"), analysis.get("complexity", "未知"))
            metadata_content.append(f"- 复杂度: {complexity}")
            metadata_content.append(
                f"- 需要代码示例: {'是' if analysis.get('requires_code_examples') else '否'}")
            metadata_content.append(f"- 技术性问题: {'是' if analysis.get('is_technical') else '否'}")
            if "explanation" in analysis:
                metadata_content.append(f"- 策略选择理由: {analysis.get('explanation')}")

        # 子查询
        if "subquestions" in metadata:
            metadata_content.append("**分解的子查询**:")
            for i, subq in enumerate(metadata["subquestions"]):
                metadata_content.append(f"- 子查询 {i + 1}: {subq}")

        # 重写查询
        if "rewritten_query" in metadata:
            metadata_content.append(f"**重写后的查询**: {metadata['rewritten_query']}")

        # HyDE假设性答案
        if "hypothetical_answer" in metadata:
            metadata_content.append(f"**假设性答案**: {metadata['hypothetical_answer']}")

        expanders_content.append(("查看检索详情", "\n".join(metadata_content)))

    # 如果有性能统计数据，准备显示内容
    if "performance" in response and response["performance"]:
        performance = response["performance"]
        perf_content = []

        # 添加总查询时间
        if "total_query_time" in performance:
            perf_content.append(f"**总查询时间**: {performance['total_query_time']:.4f}秒")

        # 添加LLM调用时间
        if "answer_generation" in performance:
            perf_content.append(f"**LLM生成回答时间**: {performance['answer_generation']:.4f}秒")

        # 添加检索时间
        if "adaptive_retrieval" in performance:
            perf_content.append(f"**检索时间**: {performance['adaptive_retrieval']:.4f}秒")

        # 添加其他关键时间
        for key, value in performance.items():
            if key not in ["total_query_time", "answer_generation", "adaptive_retrieval"]:
                # 格式化键名以更易读
                readable_key = key.replace("_", " ").title()
                perf_content.append(f"**{readable_key}**: {value:.4f}秒")

        expanders_content.append(("查看性能统计", "\n".join(perf_content)))

    return expanders_content


# 显示消息及其扩展器的函数
def display_message(message: Dict[str, Any]):
    """
    显示单个消息及其扩展器内容

    Args:
        message: 消息字典，包含角色、内容和可能的元数据
    """
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 如果是助手消息且包含附加信息
        if message["role"] == "assistant":
            # 创建一个包含所有可能的扩展器信息的临时响应对象
            response_data = {
                "source_documents": message.get("source_documents", []),
                "retrieval_metadata": message.get("retrieval_metadata", {}),
                "performance": message.get("performance", {})
            }

            # 生成并显示扩展器
            for title, content in generate_expander_content(response_data):
                with st.expander(title):
                    st.markdown(content)


# 处理问题提交的函数
def handle_question_submission(question):
    """处理问题提交逻辑，添加到历史并查询API"""
    # 获取侧边栏中的高级RAG设置
    use_advanced_rag = st.session_state.get("use_advanced_rag", True)

    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": question})

    # 显示助手思考中状态
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("思考中...")

        # 调用API
        response = query_api(question, advanced_rag=use_advanced_rag)

        if response:
            # 更新消息
            answer = response.get("answer", "抱歉，我无法处理您的请求。")
            message_placeholder.markdown(answer)

            # 生成并显示扩展器
            for title, content in generate_expander_content(response):
                with st.expander(title):
                    st.markdown(content)

            # 添加助手消息到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "source_documents": response.get("source_documents", []),
                "retrieval_metadata": response.get("retrieval_metadata", {}),
                "performance": response.get("performance", {})
            })
        else:
            message_placeholder.markdown("抱歉，处理您的请求时出现错误。")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "抱歉，处理您的请求时出现错误。"
            })

    # 强制刷新页面
    st.rerun()


# 添加标题
st.title("🧩 公司规范与代码知识库助手")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")

    # 添加高级RAG设置选项
    if "use_advanced_rag" not in st.session_state:
        st.session_state.use_advanced_rag = True

    st.session_state.use_advanced_rag = st.checkbox("使用高级RAG功能", value=st.session_state.use_advanced_rag)

    if st.button("清除对话历史"):
        st.session_state.messages = []
        # 调用API清除服务器端历史
        query_api("", clear_history=True)
        st.success("已清除对话历史")

    st.markdown("---")
    st.markdown("### 🔍 提示示例")
    example_questions = [
        "FastAPI的主要特点是什么？",
        "Black格式化工具的默认行长度是多少？",
        "FastAPI中如何定义路径参数？",
        "如何在Requests中设置请求超时？",
        "如何结合Requests和异步编程实现高效的并发API请求处理？"
    ]

    for q in example_questions:
        if st.button(q):
            # 触发问题提交处理
            handle_question_submission(q)

# 显示聊天历史
for message in st.session_state.messages:
    display_message(message)

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 处理用户输入的问题
    handle_question_submission(prompt)