"""
模型下载脚本 —— 从 ModelScope 下载嵌入模型到本地 ./models/ 目录。
只需运行一次，后续 rag_system.py 会自动使用本地模型。
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "iic" / "nlp_corom_sentence-embedding_chinese-base"


def download_model():
    """从 ModelScope 下载中文嵌入模型"""
    print("正在从 ModelScope 下载嵌入模型 ...")
    print(f"目标路径: {MODEL_DIR}")
    print("模型大小约 390MB，请耐心等待 ...\n")

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[错误] 请先安装 modelscope: pip install modelscope")
        sys.exit(1)

    try:
        local_path = snapshot_download(
            "iic/nlp_corom_sentence-embedding_chinese-base",
            cache_dir=str(BASE_DIR / "models"),
        )
        print(f"\n✓ 模型下载完成: {local_path}")

        # 验证关键文件
        bin_file = MODEL_DIR / "pytorch_model.bin"
        if bin_file.exists():
            size_mb = bin_file.stat().st_size / 1024 / 1024
            print(f"✓ pytorch_model.bin ({size_mb:.0f} MB)")
        else:
            print("[警告] 未找到 pytorch_model.bin，模型可能未完整下载")

    except Exception as e:
        print(f"[错误] 下载失败: {e}")
        print("请检查网络连接后重试")
        sys.exit(1)


def main():
    # 检查是否已存在
    bin_file = MODEL_DIR / "pytorch_model.bin"
    if bin_file.exists():
        size_mb = bin_file.stat().st_size / 1024 / 1024
        print(f"模型已存在 ({size_mb:.0f} MB)，无需重复下载。")
        print(f"路径: {MODEL_DIR}")
        return

    download_model()


if __name__ == "__main__":
    main()
