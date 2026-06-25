"""
模型下载脚本 —— 从 ModelScope / HuggingFace 下载所需模型到本地。
只需运行一次，后续 rag_system.py 会自动使用本地模型。

下载内容：
  1. 嵌入模型（~390MB）：用于文档向量化
  2. Reranker 模型（~1GB）：用于检索结果精排
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_DIR = BASE_DIR / "models" / "iic" / "nlp_corom_sentence-embedding_chinese-base"
RERANKER_DIR = BASE_DIR / "models" / "BAAI" / "bge-reranker-base"

try:
    from modelscope import snapshot_download
except ImportError:
    print("[错误] 请先安装 modelscope: pip install modelscope")
    sys.exit(1)


def download_embedding_model():
    """从 ModelScope 下载中文嵌入模型（~390MB）"""
    target = EMBEDDING_DIR
    bin_file = target / "pytorch_model.bin"
    if bin_file.exists():
        size_mb = bin_file.stat().st_size / 1024 / 1024
        print(f"✓ 嵌入模型已存在 ({size_mb:.0f} MB)，跳过下载")
        return

    print("正在从 ModelScope 下载嵌入模型（~390MB）...")
    print(f"目标路径: {target}\n")
    try:
        local_path = snapshot_download(
            "iic/nlp_corom_sentence-embedding_chinese-base",
            cache_dir=str(BASE_DIR / "models"),
        )
        print(f"✓ 嵌入模型下载完成: {local_path}")
        if bin_file.exists():
            print(f"✓ pytorch_model.bin ({bin_file.stat().st_size / 1024 / 1024:.0f} MB)")
    except Exception as e:
        print(f"[错误] 下载失败: {e}")
        sys.exit(1)


def download_reranker_model():
    """
    下载 Cross-Encoder 重排序模型（~1GB）。
    优先 ModelScope，回退 HuggingFace 镜像。
    """
    target = RERANKER_DIR

    # 检测是否已存在
    if target.exists() and any(target.iterdir()):
        print(f"✓ Reranker 模型已存在，跳过下载")
        print(f"  路径: {target}")
        return

    print("正在下载 Reranker 模型（~1GB）...")
    print(f"目标路径: {target}\n")

    # 方案 A：ModelScope（国内快）
    try:
        local_path = snapshot_download(
            "iic/bge-reranker-base",
            cache_dir=str(BASE_DIR / "models"),
        )
        # ModelScope 下载的模型缺少 sentence_transformers 配置文件，手动创建
        cfg_file = target / "config_sentence_transformers.json"
        if not cfg_file.exists():
            import json
            cfg_file.write_text(json.dumps({"model_type": "CrossEncoder"}), encoding="utf-8")
            print("        (已补充 config_sentence_transformers.json)")

        print(f"✓ Reranker 模型下载完成: {local_path}")
        return
    except Exception as e:
        print(f"  ModelScope 不可用 ({e})，改用 HuggingFace 镜像 ...")

    # 方案 B：HuggingFace 镜像
    try:
        import httpx
        from huggingface_hub import snapshot_download as hf_snapshot

        # 配置 SSL + 镜像
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        try:
            from huggingface_hub import set_client_factory, close_session
            close_session()
            set_client_factory(lambda: httpx.Client(verify=False))
        except Exception:
            pass

        local_path = hf_snapshot(
            "BAAI/bge-reranker-base",
            cache_dir=str(BASE_DIR / "models" / "huggingface"),
        )
        print(f"✓ Reranker 模型下载完成 (HF): {local_path}")
    except Exception as e:
        print(f"[警告] Reranker 下载失败: {e}")
        print("  RAG 系统仍可运行，但 Reranker 功能将不可用。")
        print("  你可手动下载 BAAI/bge-reranker-base 放到 models/ 目录。")


def main():
    print("=" * 60)
    print("  RAG 模型下载工具")
    print("=" * 60)

    print("\n[1/2] 嵌入模型")
    download_embedding_model()

    print("\n[2/2] Reranker 模型")
    download_reranker_model()

    print("\n" + "=" * 60)
    print("  模型下载完成！现在可以启动 RAG 系统了。")
    print("=" * 60)


if __name__ == "__main__":
    main()
