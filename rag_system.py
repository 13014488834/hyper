"""
RAG 问答系统 —— CLI 模式

用法:
    python rag_system.py                        # 仅使用 knowledge.txt
    python rag_system.py --pdf report.pdf       # 额外加载 PDF
    python rag_system.py --pdf a.pdf b.pdf      # 加载多个 PDF
    python rag_system.py --txt extra.txt        # 额外加载 TXT
    python rag_system.py --rebuild              # 强制重建向量库
"""

import argparse
import sys
from pathlib import Path

# 核心管道（SSL 补丁、嵌入、向量库、LLM、RAG 链）
from rag_core import (
    BASE_DIR,
    CHROMA_PERSIST_DIR,
    TOP_K,
    LLM_MODEL,
    LLM_TEMPERATURE,
    load_api_key,
    split_documents,
    load_embeddings,
    build_or_load_vectorstore,
    init_llm,
    build_rag_chain,
    query_rag,
    format_sources,
    print_step,
    load_reranker,
    RERANK_TOP_K,
)

# 文件加载
from pdf_loader import (
    load_pdf,
    load_pdfs,
    load_text_file,
    merge_knowledge_sources,
)

# 默认知识库路径
KNOWLEDGE_PATH = BASE_DIR / "knowledge.txt"


# ====== 命令行交互循环 ======
def interactive_qa(rag_chain, retriever) -> None:
    """命令行交互循环：输入问题 → 检索 → 生成 → 显示答案和来源。"""
    print("\n" + "=" * 70)
    print("  RAG 问答系统已就绪！")
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

        # 检索 + 生成
        print(f"\n⏳ 正在检索 Top-{TOP_K} 相关文档 ...")
        try:
            answer, docs = query_rag(question, rag_chain, retriever)
        except Exception as e:
            print(f"[错误] 问答失败: {e}")
            continue

        if not docs:
            print("⚠ 未检索到相关文档。请丰富知识库后重建向量库（--rebuild）。")
            continue

        # 检索结果预览
        sources = format_sources(docs, max_length=200)
        print(f"\n📚 检索到 {len(sources)} 个相关文档片段:")
        for s in sources:
            print(f"   [{s['index']}] {s['snippet']}...")

        # 答案
        print("\n" + "-" * 60)
        print("🤖 回答:")
        print(answer)
        print("-" * 60)

        # 引用来源
        print("\n📖 引用来源:")
        for s in sources:
            print(f"   来源 [{s['index']}]: {s['snippet'][:150]}...")


# ====== 主入口 ======
def main():
    parser = argparse.ArgumentParser(
        description="RAG 问答系统 —— 基于本地知识库的智能问答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python rag_system.py                          # 默认知识库
  python rag_system.py --pdf contract.pdf       # 加载 PDF
  python rag_system.py --pdf a.pdf b.pdf        # 多个 PDF
  python rag_system.py --txt extra.txt          # 额外 TXT
  python rag_system.py --rebuild                # 强制重建向量库
        """,
    )
    parser.add_argument("--pdf", nargs="+", help="要加载的 PDF 文件路径")
    parser.add_argument("--txt", nargs="+", help="要额外加载的 TXT 文件路径")
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量库")
    parser.add_argument("--no-hybrid", action="store_true", help="禁用混合检索（仅用语义检索）")
    parser.add_argument("--no-reranker", action="store_true", help="禁用 Reranker 精排")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG 问答系统 启动中 ...")
    print("=" * 60)

    # ---- 收集知识来源 ----
    knowledge_texts = []

    # 1) 默认 knowledge.txt
    if KNOWLEDGE_PATH.exists():
        try:
            knowledge_texts.append(load_text_file(KNOWLEDGE_PATH))
        except Exception as e:
            print(f"       ⚠ 加载默认知识库失败: {e}")

    # 2) --pdf 参数
    if args.pdf:
        print_step(1, f"加载 PDF 文件 ({len(args.pdf)} 个)")
        pdf_paths = [Path(p).resolve() for p in args.pdf]
        pdf_text = load_pdfs(pdf_paths)
        if pdf_text.strip():
            knowledge_texts.append(pdf_text)

    # 3) --txt 参数
    if args.txt:
        print_step(1, f"加载额外 TXT 文件 ({len(args.txt)} 个)")
        for p in args.txt:
            try:
                knowledge_texts.append(load_text_file(Path(p).resolve()))
            except Exception as e:
                print(f"       ⚠ 跳过 {p}: {e}")

    # 合并
    full_text = merge_knowledge_sources(*knowledge_texts)
    if not full_text.strip():
        print("[错误] 没有任何知识内容，请准备 knowledge.txt 或通过 --pdf/--txt 指定文件")
        sys.exit(1)

    print(f"\n       ✓ 知识库总长度: {len(full_text)} 字符")

    # ---- 标准 RAG 管道 ----
    api_key = load_api_key()
    print(f"       ✓ 已读取 API Key ({api_key[:8]}...)")

    documents = split_documents(full_text)
    print(f"       ✓ 文档已切分为 {len(documents)} 个片段")

    print_step(3, "加载嵌入模型")
    embeddings = load_embeddings()

    print_step(4, "构建/加载向量数据库")
    vectorstore = build_or_load_vectorstore(
        documents, embeddings,
        persist_dir=CHROMA_PERSIST_DIR,
        force_rebuild=args.rebuild,
    )

    print_step(5, "初始化 LLM")
    llm = init_llm(api_key)

    print_step(6, "构建 RAG 问答链")

    # 混合检索 / Reranker 配置
    use_hybrid = not args.no_hybrid
    use_reranker = not args.no_reranker
    reranker = None

    if use_reranker:
        try:
            reranker = load_reranker()
        except Exception as e:
            print(f"       ⚠ Reranker 加载失败，将跳过精排: {e}")
            use_reranker = False

    rag_chain, retriever = build_rag_chain(
        vectorstore, llm,
        documents=documents,        # BM25 需要原始文档
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        reranker=reranker,
    )

    # 进入交互
    interactive_qa(rag_chain, retriever)


if __name__ == "__main__":
    main()
