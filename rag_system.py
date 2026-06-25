"""
RAG 问答系统 —— 基于 LangChain + DeepSeek + Chroma + HuggingFace 嵌入

功能：
  1. 读取 knowledge.txt 作为本地知识库
  2. RecursiveCharacterTextSplitter 切分文档 (chunk_size=500, overlap=50)
  3. HuggingFaceEmbeddings 向量化，存入 Chroma 并持久化到 ./chroma_db
  4. ChatDeepSeek 接入 DeepSeek API（从 .env 读取 key），生成最终答案
  5. 命令行交互问答，检索 Top-3 文档片段，显示答案与引用来源
  6. 嵌入模型下载慢时自动切换为 shibing624/text2vec-base-chinese
"""

# ====== 在所有 import 之前：禁用 SSL 验证 + 修复 Windows 编码 ======
import os as _os
import sys as _sys

# 1) 环境变量（huggingface_hub / requests / httpx 通用）
_os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
_os.environ["CURL_CA_BUNDLE"] = ""
_os.environ["REQUESTS_CA_BUNDLE"] = ""
# ★ 国内用户：使用 hf-mirror.com 镜像下载模型（避免 huggingface.co 被墙）
if not _os.environ.get("HF_ENDPOINT"):
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2) Python 标准库 ssl —— 全局禁用证书验证
import ssl as _ssl
try:
    _ssl._create_default_https_context = _ssl._create_unverified_context
except AttributeError:
    pass

# 3) urllib3 —— 禁用 SSL 警告
try:
    import urllib3 as _urllib3
    _urllib3.disable_warnings(_urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# 4) Windows 控制台编码修复（GBK → UTF-8）
if _sys.platform == "win32":
    import io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")
    _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding="utf-8", errors="replace")
    _os.environ["PYTHONIOENCODING"] = "utf-8"

# 清理临时引用
del _os, _sys, _ssl, _urllib3
try:
    del _io
except NameError:
    pass

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# 0. 环境与依赖导入（带友好的错误提示）
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 1. 配置与常量
# ---------------------------------------------------------------------------
# 项目根目录（兼容 Windows 路径）
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
KNOWLEDGE_PATH = BASE_DIR / "knowledge.txt"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")

# 文档切分参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索参数
TOP_K = 3

# 嵌入模型 —— 主模型 & 备用模型
# 优先使用 ModelScope 下载的本地模型（国内网络快），其次走 HuggingFace 镜像
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


# ---------------------------------------------------------------------------
# 2. 工具函数
# ---------------------------------------------------------------------------
def print_step(step_num: int, message: str) -> None:
    """打印带编号的进度步骤"""
    print(f"\n[步骤 {step_num}] {message} ...")


def load_env() -> str:
    """
    加载 .env 文件并返回 DEEPSEEK_API_KEY。
    若未配置则退出程序。
    """
    if not ENV_PATH.exists():
        print(f"[错误] 未找到 .env 文件，请确认文件路径: {ENV_PATH}")
        print("         .env 文件应包含: DEEPSEEK_API_KEY=你的密钥")
        sys.exit(1)

    load_dotenv(dotenv_path=str(ENV_PATH))
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("[错误] .env 文件中未配置 DEEPSEEK_API_KEY 或值为空")
        print("         请在 .env 文件中添加: DEEPSEEK_API_KEY=你的密钥")
        sys.exit(1)
    return api_key


def load_knowledge(file_path: Path) -> str:
    """
    读取知识库文件，返回文本内容。
    文件不存在或为空时退出。
    """
    if not file_path.exists():
        print(f"[错误] 知识库文件不存在: {file_path}")
        print("         请在该路径下创建 knowledge.txt 并添加知识内容")
        sys.exit(1)

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Windows 下某些文件可能是 GBK 编码
        content = file_path.read_text(encoding="gbk")

    if not content.strip():
        print(f"[错误] 知识库文件为空: {file_path}")
        print("         请在 knowledge.txt 中添加知识内容后重新运行")
        sys.exit(1)

    print(f"       ✓ 已读取知识库，共 {len(content)} 个字符")
    return content


# ---------------------------------------------------------------------------
# 3. 文档切分
# ---------------------------------------------------------------------------
def split_documents(text: str) -> list:
    """
    使用 RecursiveCharacterTextSplitter 将文本切分为文档片段。

    参数:
        text: 原始文本字符串

    返回:
        List[Document]: 切分后的 Document 列表
    """
    print_step(2, "正在切分文档")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # 每个片段最大 500 字符
        chunk_overlap=CHUNK_OVERLAP,    # 相邻片段重叠 50 字符，保持语义连贯
        separators=["\n\n", "\n", "。", ".", "！", "？", "，", " ", ""],  # 优先按自然段落/句子切分
        length_function=len,
        is_separator_regex=False,
    )
    documents = splitter.create_documents([text])
    print(f"       ✓ 文档已切分为 {len(documents)} 个片段 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return documents


# ---------------------------------------------------------------------------
# 4. 嵌入模型加载（带主模型/备用模型自动切换）
# ---------------------------------------------------------------------------
def _configure_hf_client_ssl() -> None:
    """
    配置 huggingface_hub 的 HTTP 客户端，禁用 SSL 证书验证。
    必须在任何模型下载之前调用！
    huggingface_hub 底层使用 httpx，此函数通过 set_client_factory
    注入一个 verify=False 的 httpx.Client，解决企业网络 SSL 证书问题。
    """
    try:
        import httpx
        from huggingface_hub import set_client_factory, close_session

        # 先关闭旧 session（如果存在）
        close_session()

        # 创建一个禁用 SSL 验证的客户端工厂
        def _create_insecure_client() -> httpx.Client:
            return httpx.Client(verify=False)

        set_client_factory(_create_insecure_client)
        print("        (已配置 HF Hub 客户端: SSL 验证已禁用)")
    except Exception as e:
        # 如果配置失败，静默忽略（让后续的模型加载自行处理）
        pass


def load_embeddings() -> HuggingFaceEmbeddings:
    """
    加载 HuggingFace 嵌入模型（三级 fallback）：
      1) 本地已下载的 ModelScope 模型（最快）
      2) HuggingFace 镜像下载 all-MiniLM-L6-v2
      3) HuggingFace 镜像下载 shibing624/text2vec-base-chinese
    SSL 验证已在模块顶层全局禁用，此处直接加载。
    """
    print_step(3, "正在加载嵌入模型")

    def _load_model(model_name: str) -> HuggingFaceEmbeddings:
        """实际加载模型"""
        is_local = os.path.isdir(model_name)
        if is_local:
            print(f"       加载本地模型: {model_name} ...")
        else:
            print(f"       从远程下载/加载模型: {model_name} ...")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # ---- 1) 优先尝试本地模型（通过 ModelScope 下载好的）----
    if os.path.isdir(LOCAL_EMBEDDING_MODEL):
        try:
            embeddings = _load_model(LOCAL_EMBEDDING_MODEL)
            print(f"       ✓ 本地嵌入模型加载成功: {LOCAL_EMBEDDING_MODEL}")
            return embeddings
        except Exception as e:
            print(f"       ⚠ 本地模型加载失败: {e}")

    # ---- 2) 尝试从 HuggingFace 下载主模型 ----
    _configure_hf_client_ssl()
    try:
        embeddings = _load_model(PRIMARY_EMBEDDING_MODEL)
        print(f"       ✓ 嵌入模型加载成功: {PRIMARY_EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        print(f"       ⚠ 主模型加载失败: {e}")

    # ---- 3) 切换备用模型 ----
    _configure_hf_client_ssl()
    print(f"       自动切换备用模型: {FALLBACK_EMBEDDING_MODEL}")
    try:
        embeddings = _load_model(FALLBACK_EMBEDDING_MODEL)
        print(f"       ✓ 备用嵌入模型加载成功: {FALLBACK_EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        print(f"[错误] 备用模型也无法加载: {e}")
        print("         请检查网络连接，或手动下载嵌入模型后重试")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 5. 向量库构建（Chroma 持久化）
# ---------------------------------------------------------------------------
def build_vectorstore(
    documents: list, embeddings: HuggingFaceEmbeddings
) -> Chroma:
    """
    将文档向量化并存入 Chroma，持久化到 ./chroma_db。
    若向量库已存在则直接加载（避免重复计算）。
    """
    print_step(4, "正在构建向量数据库")

    # 如果已存在持久化的向量库，直接加载
    if os.path.isdir(CHROMA_PERSIST_DIR) and any(
        name.endswith(".parquet") or name.endswith(".bin") or name == "chroma.sqlite3"
        for name in os.listdir(CHROMA_PERSIST_DIR)
    ):
        print(f"       检测到已有向量库，从 {CHROMA_PERSIST_DIR} 加载 ...")
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
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
            persist_directory=CHROMA_PERSIST_DIR,
        )
        # Chroma 在某些版本中 from_documents 自动持久化，这里再显式确认一次
        print(f"       ✓ 向量库已构建并持久化到 {CHROMA_PERSIST_DIR}")
        print(f"       ✓ 共 {vectorstore._collection.count()} 条向量记录")
        return vectorstore
    except Exception as e:
        print(f"[错误] 构建向量库失败: {e}")
        print("         可能是 Chroma 依赖问题，请确保已安装 chromadb")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 6. DeepSeek LLM 初始化
# ---------------------------------------------------------------------------
def init_llm(api_key: str) -> ChatDeepSeek:
    """
    初始化 ChatDeepSeek 实例。
    """
    print_step(5, "正在初始化 DeepSeek LLM")
    try:
        llm = ChatDeepSeek(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            api_key=api_key,
            # 设置超时，避免网络问题导致长时间卡住
            timeout=60,
            max_retries=2,
        )
        print(f"       ✓ LLM 初始化成功 (model={LLM_MODEL}, temperature={LLM_TEMPERATURE})")
        return llm
    except Exception as e:
        print(f"[错误] 初始化 DeepSeek LLM 失败: {e}")
        print("         请检查 API Key 是否正确以及网络连接")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 7. RAG 链构建
# ---------------------------------------------------------------------------
def build_rag_chain(vectorstore: Chroma, llm: ChatDeepSeek):
    """
    构建 RAG 链：检索 → 组装提示词 → LLM 生成答案。

    返回:
        rag_chain: 可调用的 RAG 链对象
        retriever:  检索器（用于获取来源文档）
    """
    print_step(6, "正在构建 RAG 问答链")

    # 检索器 —— 返回 Top-K 个最相关文档
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    # 提示词模板
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 格式化检索到的文档为上下文字符串
    def format_docs(docs) -> str:
        """将检索到的文档格式化为带编号的上下文字符串"""
        formatted = []
        for i, doc in enumerate(docs, start=1):
            content = doc.page_content.replace("\n", " ").strip()
            formatted.append(f"【文档片段 {i}】\n{content}")
        return "\n\n".join(formatted)

    # 构建 RAG 链（使用 LCEL）
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("       ✓ RAG 链构建完成")
    return rag_chain, retriever


# ---------------------------------------------------------------------------
# 8. 命令行交互问答
# ---------------------------------------------------------------------------
def interactive_qa(rag_chain, retriever) -> None:
    """
    命令行交互循环：用户输入问题 → 检索 → 生成答案 → 显示结果。
    输入 'quit' / 'exit' / 'q' 退出。
    """
    print("\n" + "=" * 70)
    print("  RAG 问答系统已就绪！")
    print(f"  知识库: {KNOWLEDGE_PATH}")
    print(f"  检索数量: Top-{TOP_K}")
    print(f"  LLM: {LLM_MODEL} (temperature={LLM_TEMPERATURE})")
    print("  输入 'quit' / 'exit' / 'q' 退出")
    print("=" * 70)

    while True:
        try:
            question = input("\n🧑 请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        # ---- 检索相关文档 ----
        print(f"\n⏳ 正在检索 Top-{TOP_K} 相关文档 ...")
        try:
            docs = retriever.invoke(question)
        except Exception as e:
            print(f"[错误] 检索失败: {e}")
            continue

        if not docs:
            print("⚠ 未检索到相关文档，无法回答。请丰富 knowledge.txt 后重新构建向量库。")
            continue

        # 显示检索到的文档片段
        print(f"\n📚 检索到 {len(docs)} 个相关文档片段:")
        for i, doc in enumerate(docs, start=1):
            snippet = doc.page_content.replace("\n", " ").strip()[:200]
            print(f"   [{i}] {snippet}...")

        # ---- 生成答案 ----
        print("\n⏳ 正在生成答案 ...")
        try:
            answer = rag_chain.invoke(question)
        except Exception as e:
            print(f"[错误] 调用 LLM 失败: {e}")
            print("         请检查 DeepSeek API Key 余额和网络连接")
            continue

        # ---- 输出结果 ----
        print("\n" + "-" * 60)
        print("🤖 回答:")
        print(answer)
        print("-" * 60)

        # 输出引用来源
        print("\n📖 引用来源:")
        for i, doc in enumerate(docs, start=1):
            snippet = doc.page_content.replace("\n", " ").strip()[:150]
            print(f"   来源 [{i}]: {snippet}...")


# ---------------------------------------------------------------------------
# 9. 主入口
# ---------------------------------------------------------------------------
def main():
    """主流程：按顺序执行各初始化步骤，最后进入交互问答循环。"""
    print("=" * 60)
    print("  RAG 问答系统 启动中 ...")
    print("=" * 60)

    # 步骤 0: 加载环境变量
    print_step(0, "加载配置")
    api_key = load_env()
    print(f"       ✓ 已读取 API Key ({api_key[:8]}...)")

    # 步骤 1: 读取知识库
    print_step(1, "读取知识库文件")
    knowledge_text = load_knowledge(KNOWLEDGE_PATH)

    # 步骤 2: 切分文档
    documents = split_documents(knowledge_text)

    # 步骤 3: 加载嵌入模型（带超时自动切换）
    embeddings = load_embeddings()

    # 步骤 4: 构建/加载向量库
    vectorstore = build_vectorstore(documents, embeddings)

    # 步骤 5: 初始化 LLM
    llm = init_llm(api_key)

    # 步骤 6: 构建 RAG 链
    rag_chain, retriever = build_rag_chain(vectorstore, llm)

    # 步骤 7: 进入交互问答
    interactive_qa(rag_chain, retriever)


if __name__ == "__main__":
    main()
