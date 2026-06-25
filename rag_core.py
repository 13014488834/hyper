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

# 混合检索参数（BM25 + 语义）
MERGE_TOP_K = 10     # 每路检索取多少候选用 RRF 融合
RRF_K = 60           # RRF 平滑参数（越大排名差异越小）

# Reranker 模型
RERANKER_MODEL = "BAAI/bge-reranker-base"
LOCAL_RERANKER_MODEL = str(BASE_DIR / "models" / "BAAI" / "bge-reranker-base")
RERANK_TOP_K = 3     # 精排后保留的文档数

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


# ====== 混合检索 ======
class HybridRetriever:
    """
    混合检索器：BM25 关键词 + 语义向量，RRF 融合排序。

    原理：
      - BM25 擅长精确关键词匹配（如人名、编号、术语）
      - 语义检索擅长理解意图和同义表达
      - RRF（Reciprocal Rank Fusion）将两路排名合并，不需要手动调权重
    """

    def __init__(self, documents: list, vectorstore: Chroma, merge_k: int = MERGE_TOP_K):
        from langchain_community.retrievers import BM25Retriever

        # 构建 BM25 索引（基于原始文档的 token 频率统计）
        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = merge_k

        # 语义检索器
        self.vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": merge_k}
        )

    def invoke(self, query: str) -> list:
        """并行检索两路，RRF 融合后返回 top-k 文档"""
        bm25_docs = self.bm25.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        return rrf_fusion(bm25_docs, vector_docs, k=RRF_K)


def rrf_fusion(
    results_a: list, results_b: list, k: int = 60, top_n: int = None
) -> list:
    """
    Reciprocal Rank Fusion —— 将两路检索结果重新排序。

    算法：
      score(doc) = Σ 1 / (k + rank_i(doc))
      其中 rank_i 是文档在第 i 路检索结果中的排名（从 1 开始），
      k 是平滑参数（默认 60），避免排名靠前的文档主导结果。

    用文档内容做去重 key，两路都出现的文档得分会叠加。
    """
    if top_n is None:
        top_n = TOP_K

    scores: dict = {}
    doc_map: dict = {}  # content_hash -> Document

    for rank, doc in enumerate(results_a, start=1):
        key = doc.page_content[:200]  # 用前 200 字符做去重特征
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_map:
            doc_map[key] = doc

    for rank, doc in enumerate(results_b, start=1):
        key = doc.page_content[:200]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in doc_map:
            doc_map[key] = doc

    # 按 RRF 融合分降序排列
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [doc_map[key] for key in sorted_keys[:top_n]]


# ====== Reranker 精排 ======
def load_reranker(model_path: str = None) -> "Reranker":
    """
    加载 Cross-Encoder 精排模型。
    优先本地 ModelScope 缓存，其次从 HuggingFace 镜像下载。
    """
    if model_path is None:
        # 检查本地路径
        if os.path.isdir(LOCAL_RERANKER_MODEL):
            model_path = LOCAL_RERANKER_MODEL
        else:
            model_path = RERANKER_MODEL

    print(f"       正在加载 Reranker 模型: {model_path} ...")
    return Reranker(model_path)


class Reranker:
    """
    Cross-Encoder 精排器。

    与 Bi-Encoder（Embedding）不同，Cross-Encoder 将 query 和 document
    拼接后送入 Transformer，做深度交叉注意力计算，因此排序精度远高于
    向量相似度，但速度较慢，适合对少量候选做二次精排。

    典型用法：粗排（BM25+语义）→ 取 Top-10 → Reranker 精排 → 取 Top-3
    """

    def __init__(self, model_name_or_path: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name_or_path)

    def rerank(self, query: str, docs: list, top_k: int = RERANK_TOP_K) -> list:
        """对文档列表重新打分排序，返回 top_k 个"""
        if len(docs) <= top_k:
            return docs

        # 构造 query-doc 对
        pairs = [[query, doc.page_content] for doc in docs]
        # Cross-Encoder 预测相关性分数
        scores = self.model.predict(pairs)

        # 分数归一化（可选，提升可读性）
        ranked = sorted(
            zip(docs, scores), key=lambda x: x[1], reverse=True
        )
        return [doc for doc, _ in ranked[:top_k]]


# ====== RAG 链构建 ======
def build_rag_chain(
    vectorstore: Chroma,
    llm: ChatDeepSeek,
    documents: list = None,
    top_k: int = TOP_K,
    prompt_template: str = RAG_PROMPT_TEMPLATE,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    reranker: "Reranker" = None,
):
    """
    构建 RAG 链：检索 → 组装提示词 → LLM 生成。
    支持混合检索（BM25 + 语义）和 Reranker 精排。

    参数:
        vectorstore: Chroma 向量库
        llm: 大模型实例
        documents: 原始文档列表（混合检索需要 BM25 索引）
        top_k: 最终返回给 LLM 的文档数
        use_hybrid: 是否启用 BM25 + 语义混合检索
        use_reranker: 是否启用 Cross-Encoder 精排
        reranker: Reranker 实例（use_reranker=True 时必传）

    返回:
        (rag_chain, retriever) —— retriever 对外暴露，供 query_rag 取来源文档
    """
    # ---- 构建检索器 ----
    if use_hybrid and documents:
        # 混合检索：BM25 + 语义 → RRF 融合
        print("       🔀 启用混合检索（BM25 + 语义向量，RRF 融合）")
        hybrid = HybridRetriever(documents, vectorstore, merge_k=MERGE_TOP_K)
        base_retriever = hybrid
    else:
        # 纯语义检索
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # ---- 叠加 Reranker ----
    if use_reranker and reranker is not None:
        print(f"       🎯 启用 Reranker 精排（Cross-Encoder，Top-{RERANK_TOP_K}）")

        # 包装为 LangChain 兼容的 retriever
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document as LCDocument

        class RerankerRetriever(BaseRetriever):
            """将 Reranker 包装为 LangChain BaseRetriever"""
            def _get_relevant_documents(self, query: str) -> list:
                # 步骤 1: 粗排 —— 混合检索取 Top-10
                docs = base_retriever.invoke(query)
                # 步骤 2: 精排 —— Cross-Encoder 重打分
                return reranker.rerank(query, docs, top_k=RERANK_TOP_K)

        retriever = RerankerRetriever()
    else:
        retriever = base_retriever

    # ---- 构建 LCEL 链 ----
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
