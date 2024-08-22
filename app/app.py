import os
import requests
import streamlit as st
from dotenv import load_dotenv

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


# 添加标题
st.title("🧩 公司规范与代码知识库助手")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []


# 添加一个辅助函数处理问题提交
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

            # 准备扩展器的内容
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
                        "decomposition": "查询分解",
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

                # 子查询
                if "subquestions" in metadata:
                    metadata_content.append("**分解的子查询**:")
                    for i, subq in enumerate(metadata["subquestions"]):
                        metadata_content.append(f"- 子查询 {i + 1}: {subq}")

                # 重写查询
                if "rewritten_query" in metadata:
                    metadata_content.append(f"**重写后的查询**: {metadata['rewritten_query']}")

                expanders_content.append(("查看检索详情", "\n".join(metadata_content)))

            # 显示扩展器
            for title, content in expanders_content:
                with st.expander(title):
                    st.markdown(content)

            # 添加助手消息到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "source_documents": response.get("source_documents", []),
                "retrieval_metadata": response.get("retrieval_metadata", {})
            })
        else:
            message_placeholder.markdown("抱歉，处理您的请求时出现错误。")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "抱歉，处理您的请求时出现错误。"
            })

    # 强制刷新页面
    st.rerun()


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
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 如果是助手消息且包含附加信息
        if message["role"] == "assistant":
            # 显示源文档
            if "source_documents" in message and message["source_documents"]:
                with st.expander("查看源文档"):
                    for i, doc in enumerate(message["source_documents"]):
                        st.markdown(f"**源 {i + 1}**: {doc.get('metadata', {}).get('source', '未知')}")
                        st.text(doc.get("page_content", "无内容"))
                        st.markdown("---")

            # 显示检索元数据
            if "retrieval_metadata" in message and message["retrieval_metadata"]:
                metadata = message["retrieval_metadata"]
                with st.expander("查看检索详情"):
                    # 检索策略
                    if "strategy_used" in metadata:
                        strategy_map = {
                            "basic": "基本检索",
                            "decomposition": "查询分解",
                            "hybrid": "混合策略"
                        }
                        strategy = strategy_map.get(metadata["strategy_used"], metadata["strategy_used"])
                        st.markdown(f"**检索策略**: {strategy}")

                    # 查询分析
                    if "analysis" in metadata:
                        analysis = metadata["analysis"]
                        st.markdown("**查询分析**:")
                        complexity_map = {
                            "simple": "简单",
                            "medium": "中等",
                            "complex": "复杂"
                        }
                        complexity = complexity_map.get(analysis.get("complexity"), analysis.get("complexity", "未知"))
                        st.markdown(f"- 复杂度: {complexity}")
                        st.markdown(f"- 需要代码示例: {'是' if analysis.get('requires_code_examples') else '否'}")
                        st.markdown(f"- 技术性问题: {'是' if analysis.get('is_technical') else '否'}")

                    # 子查询
                    if "subquestions" in metadata:
                        st.markdown("**分解的子查询**:")
                        for i, subq in enumerate(metadata["subquestions"]):
                            st.markdown(f"- 子查询 {i + 1}: {subq}")

                    # 重写查询
                    if "rewritten_query" in metadata:
                        st.markdown(f"**重写后的查询**: {metadata['rewritten_query']}")

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 处理用户输入的问题
    handle_question_submission(prompt)