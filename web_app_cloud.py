"""
RAG 问答系统 —— 云端版（Streamlit Cloud 部署）

与本地版的区别：
  - 嵌入模型：使用 HuggingFace 免费推理 API（无需下载 390MB 本地模型）
  - Reranker：去掉（云端无需 1GB 本地模型）
  - 向量库：内存模式（轻量，每次重启自动重建）
  - 其它功能不变：上传 PDF/TXT、混合检索、聊天问答

部署方式：
  1. 推送到 GitHub
  2. 在 https://streamlit.io/cloud 连接仓库
  3. Main file path 填: web_app_cloud.py
  4. 添加 Secret: DEEPSEEK_API_KEY = sk-xxx
"""

import os
import hashlib
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="RAG 知识库问答",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== 核心模块（复用本地版的 pdf_loader 和检索逻辑）======
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever

from pdf_loader import load_pdf_from_bytes, load_text_file, merge_knowledge_sources

# ====== 配置常量 ======
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "knowledge.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
MERGE_TOP_K = 10
RRF_K = 60

# HuggingFace 免费推理 API 的嵌入模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM 参数
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048

# RAG 提示词
RAG_PROMPT_TEMPLATE = """你是一个基于知识库的智能问答助手。请严格根据以下提供的参考文档片段来回答问题。
如果参考文档中没有足够的信息，请明确说"根据现有资料，我无法回答此问题"，不要编造答案。

{context}

问题：{question}

请用中文回答，并在回答末尾列出你所引用的【文档片段编号】："""


# ====== RRF 融合 ======
def rrf_fusion(results_a: list, results_b: list, k: int = 60, top_n: int = 3) -> list:
    """Reciprocal Rank Fusion —— 将 BM25 和语义检索结果合并去重排序"""
    scores: dict = {}
    doc_map: dict = {}
    for rank, doc in enumerate(results_a, start=1):
        key = doc.page_content[:200]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_map:
            doc_map[key] = doc
    for rank, doc in enumerate(results_b, start=1):
        key = doc.page_content[:200]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_map:
            doc_map[key] = doc
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


# ====== 缓存资源（Streamlit Cloud 使用 @st.cache_resource）======
@st.cache_resource
def get_embeddings():
    """缓存嵌入模型（HuggingFace 免费推理 API，无需本地模型）"""
    print("  ✓ 使用 HuggingFace 推理 API 做嵌入（无需本地模型）")
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=os.getenv("HF_TOKEN", None),  # 可选
    )


@st.cache_resource
def get_llm(_api_key: str, _temperature: float):
    """缓存 LLM 实例"""
    return ChatDeepSeek(
        model=LLM_MODEL,
        temperature=_temperature,
        max_tokens=LLM_MAX_TOKENS,
        api_key=_api_key,
        timeout=60,
        max_retries=2,
    )


@st.cache_resource
def build_system(_docs_hash: str, _top_k: int):
    """根据当前知识文本构建向量库和检索器"""
    text = st.session_state.get("knowledge_text", "")
    if not text.strip():
        return None, None

    # 切分文档
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", "！", "？", "，", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    documents = splitter.create_documents([text])

    # 嵌入 + 向量库（内存模式，不持久化）
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    # 混合检索器：BM25 + 语义
    doc_count = vectorstore._collection.count()
    print(f"  ✓ 向量库就绪: {doc_count} 个文档片段")

    class HybridRetriever:
        def __init__(self, docs, vs, merge_k=10):
            self.bm25 = BM25Retriever.from_documents(docs)
            self.bm25.k = merge_k
            self.vector_retriever = vs.as_retriever(search_kwargs={"k": merge_k})

        def invoke(self, query):
            bm25_docs = self.bm25.invoke(query)
            vector_docs = self.vector_retriever.invoke(query)
            merged = rrf_fusion(bm25_docs, vector_docs, top_n=_top_k)
            return merged

    hybrid = HybridRetriever(documents, vectorstore)

    # 构建 LCEL 链
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    def format_docs(docs) -> str:
        formatted = []
        for i, doc in enumerate(docs, start=1):
            content = doc.page_content.replace("\n", " ").strip()
            formatted.append(f"【文档片段 {i}】\n{content}")
        return "\n\n".join(formatted)

    def retrieve_and_generate(question):
        docs = hybrid.invoke(question)
        context = format_docs(docs)
        return {"context": context, "question": question, "docs": docs}

    return hybrid, vectorstore


# ====== 会话状态初始化 ======
def init_session_state():
    defaults = {
        "messages": [],
        "knowledge_text": "",
        "loaded_sources": [],
        "system_ready": False,
        "retriever": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ====== 上传处理 ======
def handle_uploads(uploaded_files) -> None:
    new_texts = []
    new_names = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.loaded_sources:
            continue
        file_bytes = uploaded_file.read()
        try:
            if uploaded_file.name.endswith(".pdf"):
                text = load_pdf_from_bytes(file_bytes, uploaded_file.name)
            elif uploaded_file.name.endswith(".txt"):
                text = file_bytes.decode("utf-8")
            else:
                st.warning(f"不支持的类型: {uploaded_file.name}")
                continue
        except Exception as e:
            st.error(f"读取 {uploaded_file.name} 失败: {e}")
            continue
        if text.strip():
            new_texts.append(f"--- 来源: {uploaded_file.name} ---\n{text}")
            new_names.append(uploaded_file.name)

    if new_names:
        if not st.session_state.loaded_sources and not st.session_state.knowledge_text:
            if KNOWLEDGE_PATH.exists():
                try:
                    dt = load_text_file(KNOWLEDGE_PATH)
                    if dt.strip():
                        new_texts.insert(0, dt)
                except Exception:
                    pass
        old = st.session_state.knowledge_text
        st.session_state.knowledge_text = merge_knowledge_sources(old, *new_texts)
        st.session_state.loaded_sources.extend(new_names)
        st.session_state.system_ready = False
        st.cache_resource.clear()
        st.success(f"已加载 {len(new_names)} 个文件")
        st.rerun()


# ====== 主界面 ======
def main():
    init_session_state()

    # ---- 侧边栏 ----
    with st.sidebar:
        st.header("📁 知识源")
        uploaded_files = st.file_uploader(
            "上传 TXT 或 PDF", type=["txt", "pdf"],
            accept_multiple_files=True,
            help="上传后自动向量化",
        )
        if uploaded_files:
            handle_uploads(uploaded_files)

        st.caption("已加载:")
        if KNOWLEDGE_PATH.exists():
            st.caption("  📄 knowledge.txt")
        for name in st.session_state.loaded_sources:
            st.caption(f"  📄 {name}")

        st.divider()
        st.header("⚙️ 设置")

        # 从 Streamlit Cloud Secrets 或 .env 读取 API Key
        try:
            default_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        except Exception:
            default_key = ""
        if not default_key:
            default_key = os.getenv("DEEPSEEK_API_KEY", "")
        api_key = st.text_input(
            "DeepSeek API Key",
            value=default_key,
            type="password",
            placeholder="sk-...",
            help="在 Streamlit Cloud 中设为 Secret，或手动输入",
        )

        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        top_k = st.slider("Top-K", 1, 10, 3)

        st.divider()
        if st.button("🔄 重建向量库", use_container_width=True):
            st.session_state.system_ready = False
            st.cache_resource.clear()
            st.rerun()
        if st.button("🗑️ 清除对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.header("📊 状态")
        if st.session_state.system_ready:
            st.success("向量库: 就绪")
        else:
            st.info("向量库: 等待初始化")
        st.caption(f"对话轮数: {len(st.session_state.messages) // 2}")

    # ---- 主区域 ----
    st.title("🤖 RAG 知识库问答")
    st.caption("云端版 · HuggingFace API 嵌入 · DeepSeek 生成 · 上传 PDF/TXT 即用")

    # ---- 初始化系统 ----
    if st.session_state.knowledge_text and not st.session_state.system_ready:
        with st.spinner("🔄 正在初始化（向量化文档 + 构建检索器）..."):
            try:
                docs_hash = hashlib.md5(
                    st.session_state.knowledge_text.encode()
                ).hexdigest()
                retriever, vectorstore = build_system(docs_hash, top_k)

                if not api_key:
                    st.error("请先输入 DeepSeek API Key")
                    return

                llm = get_llm(_api_key=api_key, _temperature=temperature)
                prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

                def format_docs(docs) -> str:
                    formatted = []
                    for i, doc in enumerate(docs, start=1):
                        content = doc.page_content.replace("\n", " ").strip()
                        formatted.append(f"【文档片段 {i}】\n{content}")
                    return "\n\n".join(formatted)

                rag_chain = (
                    {"context": retriever.invoke | format_docs,
                     "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.system_ready = True
                st.rerun()
            except Exception as e:
                st.error(f"初始化失败: {e}")
                if "401" in str(e) or "Unauthorized" in str(e):
                    st.info("💡 可能是 HuggingFace API 需要 Token。在 Streamlit Cloud 中添加 HF_TOKEN Secret。")
                return

    # ---- 聊天记录 ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📖 引用来源"):
                    for src in msg["sources"]:
                        st.caption(f"**[{src['index']}]** {src['snippet']}...")

    # ---- 聊天输入 ----
    if prompt := st.chat_input("请输入问题..."):
        if not st.session_state.system_ready:
            st.error("请先上传知识文件")
            return
        if not api_key:
            st.error("请输入 API Key")
            return

        st.session_state.messages.append({
            "role": "user", "content": prompt, "sources": [],
        })

        with st.chat_message("assistant"):
            with st.spinner("⏳ 检索 + 生成中..."):
                try:
                    retriever = st.session_state.retriever
                    docs = retriever.invoke(prompt)
                    answer = st.session_state.rag_chain.invoke(prompt)

                    sources = []
                    for i, doc in enumerate(docs, start=1):
                        snippet = doc.page_content.replace("\n", " ").strip()[:200]
                        sources.append({"index": i, "snippet": snippet})

                    st.markdown(answer)
                    if sources:
                        with st.expander("📖 引用来源"):
                            for src in sources:
                                st.caption(f"**[{src['index']}]** {src['snippet']}...")
                except Exception as e:
                    answer = f"❌ 失败: {e}"
                    sources = []
                    st.error(answer)

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources,
        })

    if not st.session_state.knowledge_text:
        st.info("👋 上传 PDF 或 TXT 文件开始吧！系统会自动加载 knowledge.txt。")


if __name__ == "__main__":
    main()
