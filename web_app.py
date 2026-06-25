"""
RAG 问答系统 —— Web 界面（Streamlit）

用法:
    streamlit run web_app.py
    打开浏览器访问 http://localhost:8501
"""

import os
import sys
import hashlib
from pathlib import Path

import streamlit as st

# ====== 页面配置（必须是第一个 Streamlit 命令）======
st.set_page_config(
    page_title="RAG 知识库问答",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== 导入核心模块 ======
from rag_core import (
    BASE_DIR,
    CHROMA_PERSIST_DIR,
    load_api_key as core_load_api_key,
    split_documents,
    load_embeddings,
    build_or_load_vectorstore,
    init_llm,
    build_rag_chain,
    query_rag,
    format_sources,
)

from pdf_loader import (
    load_pdf_from_bytes,
    load_text_file,
    merge_knowledge_sources,
)

# 默认知识库
KNOWLEDGE_PATH = BASE_DIR / "knowledge.txt"


# ====== 缓存资源（跨 Streamlit rerun 保持） ======
@st.cache_resource
def get_embeddings():
    """缓存嵌入模型，只加载一次"""
    return load_embeddings()


@st.cache_resource
def get_llm(_api_key: str, _temperature: float):
    """缓存 LLM 实例，API Key 或 temperature 变化时重建"""
    return init_llm(api_key=_api_key, temperature=_temperature)


@st.cache_resource
def get_vectorstore(_docs_hash: str, _persist_dir: str, _force_rebuild: bool):
    """
    缓存向量库。_docs_hash 是知识文本的 MD5，内容变化时自动重建。
    """
    # 从 session_state 获取当前知识文本
    text = st.session_state.get("knowledge_text", "")
    if not text.strip():
        return None

    documents = split_documents(text)
    embeddings = get_embeddings()
    return build_or_load_vectorstore(
        documents, embeddings,
        persist_dir=_persist_dir,
        force_rebuild=_force_rebuild,
    )


# ====== 会话状态初始化 ======
def init_session_state():
    """初始化 Streamlit session_state 中的默认值"""
    defaults = {
        "messages": [],              # 聊天记录: [{"role", "content", "sources"}]
        "knowledge_text": "",        # 当前知识库完整文本
        "loaded_sources": [],        # 已加载的数据源名称列表
        "rebuild_counter": 0,        # 强制重建向量库的计数器
        "system_ready": False,       # 系统是否已就绪
        "rag_chain": None,
        "retriever": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ====== 侧边栏：知识源管理 ======
def render_sidebar():
    """渲染侧边栏（文件上传、设置、状态）"""
    with st.sidebar:
        st.header("📁 知识源")

        # 文件上传
        uploaded_files = st.file_uploader(
            "上传 TXT 或 PDF 文件",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="file_uploader",
            help="支持 .txt 和 .pdf 文件，上传后自动合并到知识库",
        )

        if uploaded_files:
            handle_uploads(uploaded_files)

        # 已加载的数据源列表
        st.caption("已加载的知识源:")
        if KNOWLEDGE_PATH.exists():
            st.caption(f"  📄 knowledge.txt（默认）")
        for name in st.session_state.loaded_sources:
            st.caption(f"  📄 {name}")
        if not st.session_state.loaded_sources and not KNOWLEDGE_PATH.exists():
            st.caption("  ⚠ 尚未加载任何知识源")

        st.divider()

        # ---- 设置 ----
        st.header("⚙️ 设置")

        # API Key —— 安全设计：不自动读取 .env，每个用户需自己填写
        # 本地开发者可勾选"从 .env 加载"来快速填入
        use_env = st.checkbox("从本地 .env 加载 Key", value=False,
                              help="勾选后自动读取服务器上的 .env 文件（仅限你自己部署时使用）")
        if use_env:
            try:
                default_key = core_load_api_key()
            except SystemExit:
                default_key = ""
                st.warning("未找到 .env 或 Key 为空")
        else:
            default_key = ""

        api_key = st.text_input(
            "DeepSeek API Key",
            value=default_key,
            type="password",
            placeholder="sk-...（每个用户需填写自己的 Key）",
            help="你的 Key 仅用于本次会话，不会被存储或共享给其他人",
        )
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key

        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            help="生成随机性：越低越确定，越高越有创意",
        )

        top_k = st.slider(
            "Top-K 检索数",
            min_value=1, max_value=10, value=3, step=1,
            help="每次检索返回的文档片段数量",
        )

        chunk_size = st.slider(
            "Chunk Size",
            min_value=200, max_value=2000, value=500, step=50,
            help="文档切分片段大小（字符数）",
        )

        st.divider()

        # ---- 操作按钮 ----
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重建向量库", use_container_width=True,
                         help="清空缓存并重新构建向量库"):
                rebuild_vectorstore(chunk_size)
        with col2:
            if st.button("🗑️ 清除对话", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        st.divider()

        # ---- 状态显示 ----
        st.header("📊 状态")
        if st.session_state.system_ready:
            st.success("向量库: 已就绪")
            try:
                vs = get_vectorstore(
                    _docs_hash=hashlib.md5(
                        st.session_state.knowledge_text.encode()
                    ).hexdigest(),
                    _persist_dir=CHROMA_PERSIST_DIR,
                    _force_rebuild=False,
                )
                if vs:
                    st.caption(f"文档片段数: {vs._collection.count()}")
            except Exception:
                pass
        else:
            st.info("向量库: 等待初始化")

        st.caption(f"对话轮数: {len(st.session_state.messages) // 2}")
        st.caption(f"已加载文件: {len(st.session_state.loaded_sources)} 个")

    return api_key, temperature, top_k, chunk_size


# ====== 上传处理 ======
def handle_uploads(uploaded_files) -> None:
    """处理上传的文件，提取文本并合并到知识库"""
    new_texts = []
    new_names = []

    for uploaded_file in uploaded_files:
        # 跳过已处理的文件
        if uploaded_file.name in st.session_state.loaded_sources:
            continue

        file_bytes = uploaded_file.read()
        try:
            if uploaded_file.name.endswith(".pdf"):
                text = load_pdf_from_bytes(file_bytes, uploaded_file.name)
            elif uploaded_file.name.endswith(".txt"):
                text = file_bytes.decode("utf-8")
            else:
                st.warning(f"不支持的文件类型: {uploaded_file.name}")
                continue
        except Exception as e:
            st.error(f"读取 {uploaded_file.name} 失败: {e}")
            continue

        if text.strip():
            new_texts.append(f"--- 来源: {uploaded_file.name} ---\n{text}")
            new_names.append(uploaded_file.name)

    if new_names:
        # 首次加载时自动加入 knowledge.txt
        if not st.session_state.loaded_sources and not st.session_state.knowledge_text:
            if KNOWLEDGE_PATH.exists():
                try:
                    default_text = load_text_file(KNOWLEDGE_PATH)
                    if default_text.strip():
                        new_texts.insert(0, default_text)
                except Exception:
                    pass

        # 合并到已有知识库
        old_text = st.session_state.knowledge_text
        st.session_state.knowledge_text = merge_knowledge_sources(old_text, *new_texts)
        st.session_state.loaded_sources.extend(new_names)
        st.session_state.system_ready = False  # 触发向量库重建

        # 刷新缓存
        st.cache_resource.clear()
        st.success(f"已加载 {len(new_names)} 个文件，正在重建向量库...")
        st.rerun()


# ====== 重建向量库 ======
def rebuild_vectorstore(chunk_size: int = 500) -> None:
    """清空并重建向量库"""
    st.session_state.rebuild_counter += 1
    st.session_state.system_ready = False
    st.cache_resource.clear()
    st.rerun()


# ====== 主界面 ======
def main():
    init_session_state()

    # ---- 侧边栏 ----
    api_key, temperature, top_k, chunk_size = render_sidebar()

    # ---- 主区域 ----
    st.title("🤖 RAG 知识库问答系统")
    st.caption("基于 LangChain + DeepSeek + Chroma | 上传 PDF/TXT 构建知识库，开始智能问答")

    # ---- 初始化 RAG 系统 ----
    if st.session_state.knowledge_text and not st.session_state.system_ready:
        with st.spinner("🔄 正在初始化 RAG 系统（加载模型 + 构建向量库）..."):
            try:
                # 计算知识文本哈希，用于缓存失效
                docs_hash = hashlib.md5(
                    st.session_state.knowledge_text.encode()
                ).hexdigest()

                embeddings = get_embeddings()
                vectorstore = get_vectorstore(
                    _docs_hash=docs_hash,
                    _persist_dir=CHROMA_PERSIST_DIR,
                    _force_rebuild=(st.session_state.rebuild_counter > 0),
                )

                if not api_key:
                    st.error("请先在侧边栏输入 DeepSeek API Key")
                    return

                llm = get_llm(_api_key=api_key, _temperature=temperature)

                # 构建 RAG 链（本地版使用纯语义检索，高级功能见 CLI）
                rag_chain, retriever = build_rag_chain(vectorstore, llm, top_k=top_k)

                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.system_ready = True
                st.rerun()
            except Exception as e:
                st.error(f"初始化失败: {e}")
                return

    # ---- 渲染聊天记录 ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📖 引用来源"):
                    for src in msg["sources"]:
                        st.caption(
                            f"**[{src['index']}]** {src['snippet']}..."
                        )

    # ---- 聊天输入 ----
    if prompt := st.chat_input("请输入你的问题..."):
        if not st.session_state.system_ready:
            st.error("⚠ 系统尚未就绪。请先上传知识文件（PDF/TXT），等待向量库初始化完成。")
            return

        if not api_key:
            st.error("⚠ 请在侧边栏输入 DeepSeek API Key")
            return

        # 添加用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "sources": [],
        })

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("⏳ 正在检索并生成答案..."):
                try:
                    answer, docs = query_rag(
                        prompt,
                        st.session_state.rag_chain,
                        st.session_state.retriever,
                    )
                    sources = format_sources(docs, max_length=200)

                    st.markdown(answer)

                    if sources:
                        with st.expander("📖 引用来源"):
                            for src in sources:
                                st.caption(
                                    f"**[{src['index']}]** {src['snippet']}..."
                                )
                except Exception as e:
                    answer = f"❌ 生成答案失败: {e}"
                    sources = []
                    st.error(answer)

        # 保存助手消息
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    # ---- 首次使用提示 ----
    if not st.session_state.knowledge_text:
        st.info(
            "👋 **欢迎使用 RAG 知识库问答系统！**\n\n"
            "请通过左侧边栏上传 PDF 或 TXT 文件开始。\n\n"
            "系统会自动加载默认的 `knowledge.txt` 文件（如果存在）。"
        )


if __name__ == "__main__":
    main()
