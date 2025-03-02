import os
import json
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


def query_api(question, clear_history=False):
    """调用API进行查询"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "clear_history": clear_history}
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

# 侧边栏
with st.sidebar:
    st.header("⚙️ 设置")

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
        "如何结合Requests和异步编程实现高效的并发API请求处理？"
    ]

    for q in example_questions:
        if st.button(q):
            st.session_state.messages.append({"role": "user", "content": q})

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 如果是助手消息且包含源文档
        if message["role"] == "assistant" and "source_documents" in message:
            with st.expander("查看源文档"):
                for i, doc in enumerate(message["source_documents"]):
                    st.markdown(f"**源 {i + 1}**: {doc.get('metadata', {}).get('source', '未知')}")
                    st.text(doc.get("page_content", "无内容"))
                    st.markdown("---")

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示助手思考中状态
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("思考中...")

        # 调用API
        response = query_api(prompt)

        if response:
            # 更新消息
            answer = response.get("answer", "抱歉，我无法处理您的请求。")
            message_placeholder.markdown(answer)

            # 如果有源文档，显示它们
            if "source_documents" in response and response["source_documents"]:
                with st.expander("查看源文档"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**源 {i + 1}**: {doc.get('metadata', {}).get('source', '未知')}")
                        st.text(doc.get("page_content", "无内容"))
                        st.markdown("---")

            # 添加助手消息到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "source_documents": response.get("source_documents", [])
            })
        else:
            message_placeholder.markdown("抱歉，处理您的请求时出现错误。")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "抱歉，处理您的请求时出现错误。"
            })