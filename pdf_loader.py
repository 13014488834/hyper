"""
PDF / TXT 文件加载模块
支持从本地文件路径和内存字节流读取文档内容。
"""

import io
from pathlib import Path
from typing import List, Union


def load_pdf(file_path: Path) -> str:
    """
    读取单个 PDF 文件，提取全部页面文本。

    参数:
        file_path: PDF 文件路径

    返回:
        提取的文本内容（页面间用空行分隔）

    异常:
        FileNotFoundError: 文件不存在
        ValueError: PDF 无可提取文本（可能是扫描件）
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("缺少 pypdf，请执行: pip install pypdf")

    if not file_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

    reader = PdfReader(str(file_path))
    if len(reader.pages) == 0:
        raise ValueError(f"PDF 文件为空: {file_path}")

    texts = []
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                texts.append(page_text.strip())
        except Exception as e:
            # 单页提取失败时跳过，继续处理后续页面
            print(f"       ⚠ PDF 第 {i+1} 页提取失败: {e}")

    if not texts:
        raise ValueError(
            f"PDF 中未找到可提取的文本 ({file_path.name})。"
            "如果是扫描件图片 PDF，需要先 OCR 识别。"
        )

    print(f"       ✓ 已从 PDF 提取文本: {file_path.name} ({len(reader.pages)} 页, {sum(len(t) for t in texts)} 字符)")
    return "\n\n".join(texts)


def load_pdf_from_bytes(data: bytes, filename: str = "uploaded.pdf") -> str:
    """
    从内存字节流读取 PDF（用于 Streamlit 等上传场景）。

    参数:
        data: PDF 文件的字节内容
        filename: 显示用的文件名（仅用于日志和错误提示）

    返回:
        提取的文本内容
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("缺少 pypdf，请执行: pip install pypdf")

    reader = PdfReader(io.BytesIO(data))
    if len(reader.pages) == 0:
        raise ValueError(f"PDF 文件为空: {filename}")

    texts = []
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                texts.append(page_text.strip())
        except Exception as e:
            print(f"       ⚠ PDF 第 {i+1} 页提取失败: {e}")

    if not texts:
        raise ValueError(f"PDF 中未找到可提取的文本 ({filename})。如果是扫描件，请先 OCR。")

    print(f"       ✓ 已从 PDF 提取文本: {filename} ({len(reader.pages)} 页, {sum(len(t) for t in texts)} 字符)")
    return "\n\n".join(texts)


def load_pdfs(file_paths: List[Path]) -> str:
    """
    批量加载多个 PDF，拼接时标注来源文件名。

    参数:
        file_paths: PDF 文件路径列表

    返回:
        合并后的文本（各 PDF 内容之间用分隔线隔开）
    """
    all_texts = []
    for path in file_paths:
        try:
            text = load_pdf(path)
            all_texts.append(f"--- 来源文件: {path.name} ---\n{text}")
        except Exception as e:
            print(f"       ⚠ 跳过 {path.name}: {e}")
    return "\n\n".join(all_texts)


def load_text_file(file_path: Path) -> str:
    """
    读取文本文件，自动检测编码（UTF-8 → GBK 回退）。

    参数:
        file_path: 文本文件路径

    返回:
        文件内容字符串

    异常:
        FileNotFoundError: 文件不存在
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Windows 下某些文件可能是 GBK 编码
        content = file_path.read_text(encoding="gbk")

    if not content.strip():
        print(f"       ⚠ 文件为空: {file_path.name}")
        return ""

    print(f"       ✓ 已读取文本文件: {file_path.name} ({len(content)} 字符)")
    return content


def merge_knowledge_sources(*texts: str) -> str:
    """
    合并多个知识文本块，用分隔线隔开。

    参数:
        *texts: 可变数量的文本块

    返回:
        合并后的文本（自动跳过空白块）
    """
    valid = [t for t in texts if t and t.strip()]
    return "\n\n---\n\n".join(valid)
