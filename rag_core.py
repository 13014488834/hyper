"""
RAG 核心模块 —— 可复用的管道组件
被 CLI（rag_system.py）和 Web（web_app.py）共同引用。

提供：
  - SSL 补丁 & Windows 编码修复（模块导入时自动执行）
  - 配置常量
  - 嵌入模型加载、向量库构建、LLM 初始化、RAG 链构建
  - 单次问答 query_rag()、来源格式化 format_sources()
"""

# ====== 在所有 import 之前：禁用 SSL 验证 + 修复 Windows 编码 ======
import os as _os
import sys as _sys

_os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
_os.environ["CURL_CA_BUNDLE"] = ""
_os.environ["REQUESTS_CA_BUNDLE"] = ""
if not _os.environ.get("HF_ENDPOINT"):
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import ssl as _ssl
try:
    _ssl._create_default_https_context = _ssl._create_unverified_context
except AttributeError:
    pass

try:
    import urllib3 as _urllib3
    _urllib3.disable_warnings(_urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

if _sys.platform == "win32":
    import io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding="utf-8", errors="replace")
    _os.environ["PYTHONIOENCODING"] = "utf-8"

del _os, _sys, _ssl, _urllib3
try:
    del _io
except NameError:
    pass

# ====== 正常 import ======
import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

# 依赖检查
try:
    from dotenv import load_dotenv
except ImportError:
    print("[错误] 缺少 python-dotenv，请执行: pip install python-dotenv")
    sys.exit(1)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("[错误] 缺少 langchain-text-splitters，请执行: pip install langchain-text-splitters")
    sys.exit(1)

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("[错误] 缺少 langchain-huggingface，请执行: pip install langchain-huggingface")
    sys.exit(1)

try:
    from langchain_chroma import Chroma
except ImportError:
    print("[错误] 缺少 langchain-chroma，请执行: pip install langchain-chroma")
    sys.exit(1)

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:
    print("[错误] 缺少 langchain-deepseek，请执行: pip install langchain-deepseek")
    sys.exit(1)

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    print("[错误] 缺少 langchain-core，请执行: pip install langchain-core")
    sys.exit(1)


# ====== 配置常量 ======
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")

# 文档切分参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索参数
TOP_K = 3

# 嵌入模型 —— 三级 fallback
LOCAL_EMBEDDING_MODEL = str(BASE_DIR / "models" / "iic" / "nlp_corom_sentence-embedding_chinese-base")
PRIMARY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

# LLM 参数
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2048

# RAG 提示词模板
RAG_PROMPT_TEMPLATE = """你是一个基于知识库的智能问答助手。请严格根据以下提供的参考文档片段来回答问题。
如果参考文档中没有足够的信息，请明确说"根据现有资料，我无法回答此问题"，不要编造答案。

{context}

问题：{question}

请用中文回答，并在回答末尾列出你所引用的【文档片段编号】："""


# ====== 工具函数 ======
def print_step(step_num: int, message: str) -> None:
    """打印带编号的进度步骤"""
    print(f"\n[步骤 {step_num}] {message} ...")


def load_api_key(env_path: Path = None) -> str:
    """
    加载 .env 文件并返回 DEEPSEEK_API_KEY。
    若未配置则退出程序。
    """
    if env_path is None:
        env_path = ENV_PATH
    if not env_path.exists():
        print(f"[错误] 未找到 .env 文件，请确认文件路径: {env_path}")
        print("         .env 文件应包含: DEEPSEEK_API_KEY=你的密钥")
        sys.exit(1)

    load_dotenv(dotenv_path=str(env_path))
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("[错误] .env 文件中未配置 DEEPSEEK_API_KEY 或值为空")
        print("         请在 .env 文件中添加: DEEPSEEK_API_KEY=你的密钥")
        sys.exit(1)
    return api_key


# ====== 文档切分 ======
def split_documents(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list:
    """使用 RecursiveCharacterTextSplitter 切分文本为 Document 列表"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", "！", "？", "，", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    documents = splitter.create_documents([text])
    return documents


# ====== 嵌入模型加载 ======
def _configure_hf_client_ssl() -> None:
    """配置 huggingface_hub HTTP 客户端，禁用 SSL 证书验证"""
    try:
        import httpx
        from huggingface_hub import set_client_factory, close_session

        close_session()

        def _create_insecure_client() -> httpx.Client:
            return httpx.Client(verify=False)

        set_client_factory(_create_insecure_client)
    except Exception:
        pass


def load_embeddings() -> HuggingFaceEmbeddings:
    """
    加载嵌入模型（三级 fallback）：
      1) 本地 ModelScope 模型
      2) all-MiniLM-L6-v2（远程）
      3) shibing624/text2vec-base-chinese（备用）
    """
    def _load_model(model_name: str) -> HuggingFaceEmbeddings:
        is_local = os.path.isdir(model_name)
        label = f"本地模型: {model_name}" if is_local else f"远程模型: {model_name}"
        print(f"       正在加载 {label} ...")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # 1) 本地模型
    if os.path.isdir(LOCAL_EMBEDDING_MODEL):
        try:
            embeddings = _load_model(LOCAL_EMBEDDING_MODEL)
            print(f"       ✓ 本地嵌入模型加载成功")
            return embeddings
        except Exception as e:
            print(f"       ⚠ 本地模型加载失败: {e}")

    # 2) 远程主模型
    _configure_hf_client_ssl()
    try:
        embeddings = _load_model(PRIMARY_EMBEDDING_MODEL)
        print(f"       ✓ 主嵌入模型加载成功: {PRIMARY_EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        print(f"       ⚠ 主模型加载失败: {e}")

    # 3) 备用模型
    _configure_hf_client_ssl()
    print(f"       自动切换备用模型: {FALLBACK_EMBEDDING_MODEL}")
    try:
        embeddings = _load_model(FALLBACK_EMBEDDING_MODEL)
        print(f"       ✓ 备用嵌入模型加载成功: {FALLBACK_EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        print(f"[错误] 备用模型也无法加载: {e}")
        print("         请检查网络连接，或运行 setup_models.py 下载本地模型")
        sys.exit(1)


# ====== 向量库构建 ======
def build_or_load_vectorstore(
    documents: list,
    embeddings: HuggingFaceEmbeddings,
    persist_dir: str = CHROMA_PERSIST_DIR,
    force_rebuild: bool = False,
) -> Chroma:
    """
    向量化文档并存入 Chroma，持久化到 persist_dir。
    若向量库已存在则直接加载（除非 force_rebuild=True）。
    """
    # 强制重建：删除旧向量库
    if force_rebuild and os.path.isdir(persist_dir):
        print(f"       强制重建：正在删除旧向量库 {persist_dir} ...")
        shutil.rmtree(persist_dir)

    # 检测已有向量库
    if os.path.isdir(persist_dir) and any(
        name.endswith(".parquet") or name.endswith(".bin") or name == "chroma.sqlite3"
        for name in os.listdir(persist_dir)
    ):
        print(f"       检测到已有向量库，从 {persist_dir} 加载 ...")
        try:
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings,
            )
            print(f"       ✓ 向量库加载成功，共 {vectorstore._collection.count()} 条记录")
            return vectorstore
        except Exception as e:
            print(f"       ⚠ 加载已有向量库失败: {e}，将重新构建 ...")

    # 新建向量库
    print(f"       正在构建新向量库（{len(documents)} 个文档）...")
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
        )
        print(f"       ✓ 向量库已构建并持久化到 {persist_dir}")
        print(f"       ✓ 共 {vectorstore._collection.count()} 条向量记录")
        return vectorstore
    except Exception as e:
        print(f"[错误] 构建向量库失败: {e}")
        sys.exit(1)


# ====== LLM 初始化 ======
def init_llm(
    api_key: str,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> ChatDeepSeek:
    """初始化 ChatDeepSeek 实例"""
    try:
        llm = ChatDeepSeek(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            timeout=60,
            max_retries=2,
        )
        print(f"       ✓ LLM 初始化成功 (model={model}, temperature={temperature})")
        return llm
    except Exception as e:
        print(f"[错误] 初始化 DeepSeek LLM 失败: {e}")
        sys.exit(1)


# ====== RAG 链构建 ======
def build_rag_chain(
    vectorstore: Chroma,
    llm: ChatDeepSeek,
    top_k: int = TOP_K,
    prompt_template: str = RAG_PROMPT_TEMPLATE,
):
    """
    构建 RAG 链：检索 → 组装提示词 → LLM 生成。
    返回 (rag_chain, retriever)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs) -> str:
        """将检索到的文档格式化为带编号的上下文字符串"""
        formatted = []
        for i, doc in enumerate(docs, start=1):
            content = doc.page_content.replace("\n", " ").strip()
            formatted.append(f"【文档片段 {i}】\n{content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("       ✓ RAG 链构建完成")
    return rag_chain, retriever


# ====== 运行时：单次问答 ======
def query_rag(question: str, rag_chain, retriever) -> Tuple[str, list]:
    """
    单次 RAG 问答：检索相关文档 + 生成答案。
    返回 (answer: str, source_docs: list[Document])
    """
    docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    return answer, docs


def format_sources(docs: list, max_length: int = 200) -> list[dict]:
    """
    将 Document 列表格式化为前端友好的来源列表。
    返回 [{"index": 1, "snippet": "...", "content": "..."}, ...]
    """
    sources = []
    for i, doc in enumerate(docs, start=1):
        content = doc.page_content.replace("\n", " ").strip()
        sources.append({
            "index": i,
            "snippet": content[:max_length],
            "content": content,
        })
    return sources
