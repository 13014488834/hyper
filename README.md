# 🤖 RAG Knowledge Base Q&A / RAG 知识库问答系统

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red.svg)](https://streamlit.io/)

A RAG (Retrieval-Augmented Generation) Q&A system powered by **LangChain + DeepSeek + Chroma**, featuring **hybrid search (BM25 + semantic) + Cross-Encoder Reranker**.

基于 **LangChain + DeepSeek + Chroma** 的知识库问答系统，支持 **混合检索（BM25 + 语义）+ Reranker 精排**。

**PDF / TXT Knowledge Base · CLI · Web Chat · Streamlit Cloud Deployment · 公网部署**

> 🌐 **English** | [中文](#中文)

---

## 🌍 Live Demo

👉 **[https://aucodhdcwzwwjmucdqhxqz.streamlit.app/](https://aucodhdcwzwwjmucdqhxqz.streamlit.app/)**

Open in browser, upload PDF/TXT, enter your DeepSeek API Key, start asking questions. No installation needed.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔀 **Hybrid Search** | BM25 (keyword) + Semantic (embedding) with RRF fusion |
| 🎯 **Reranker** | Cross-Encoder re-ranks top results for precision |
| 📁 **File Upload** | PDF + TXT, auto-chunked and vectorized |
| 💬 **Web Chat** | Streamlit web interface with chat history |
| ⌨️ **CLI Mode** | Command-line interface for scripting |
| 🌐 **Cloud Deploy** | Free hosting on Streamlit Cloud |

## 🚀 Quick Start

### Online (no install)

Open the [live demo](https://aucodhdcwzwwjmucdqhxqz.streamlit.app/), paste your DeepSeek key, upload a PDF, ask questions.

### Local (full features: Hybrid + Reranker)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download models (~1.4GB, once)
python setup_models.py

# 3. Setup API key
#    Create .env: DEEPSEEK_API_KEY=sk-your-key

# 4. Launch
streamlit run web_app.py
# → http://localhost:8501
```

### CLI

```bash
python rag_system.py                              # default knowledge.txt
python rag_system.py --pdf contract.pdf            # with PDF
python rag_system.py --rebuild                     # force rebuild index
python rag_system.py --no-hybrid --no-reranker     # turn off extras
```

## 🏗️ Architecture

| Component | Role |
|-----------|------|
| LangChain | RAG pipeline orchestration |
| DeepSeek API | LLM answer generation |
| Chroma | Vector database (semantic) |
| BM25 | Keyword retrieval |
| RRF | Reciprocal Rank Fusion |
| Cross-Encoder | Reranker (re-rank top-K) |
| Streamlit | Web UI |
| pypdf | PDF text extraction |

## 📁 Project Structure

```
├── rag_core.py          # Core pipeline (embedding, vector store, LLM, hybrid, reranker)
├── pdf_loader.py        # PDF/TXT file loader
├── rag_system.py        # CLI entry point
├── web_app.py           # Local web UI (full features)
├── web_app_cloud.py     # Cloud web UI (Streamlit Cloud)
├── setup_models.py      # Model download script (embedding + reranker)
├── knowledge.txt        # Default knowledge base
└── requirements.txt     # Dependencies
```

---

<a name="中文"></a>

## 📖 中文说明

### 使用方式

同上 Quick Start，详见上方。

### 常见问题

**Q: DeepSeek API 返回 401 或鉴权失败？**
A: 三种可能：
1. Key 格式不对——正确格式是 `sk-` 开头 + 一串字母数字
2. Key 已过期或余额用完——去 https://platform.deepseek.com 查看
3. `.env` 文件里有多余空格或引号

**Q: 嵌入模型下载失败？**
A: `setup_models.py` 默认走 ModelScope（国内快）。如果失败，手动切换 HuggingFace 镜像：
```bash
# Windows: set HF_ENDPOINT=https://hf-mirror.com
# Mac/Linux: export HF_ENDPOINT=https://hf-mirror.com
python setup_models.py
```

**Q: 云端版和本地版有什么区别？**

| | 本地版 | 云端版 |
|---|---|---|
| 嵌入模型 | 本地 390MB（免费无限次） | 云端自动下载 80MB |
| 混合检索 | ✅ BM25 + 语义 | ✅ BM25 + 语义 |
| Reranker | ✅ 本地 1GB 模型 | ❌ 未启用 |
| 启动方式 | `streamlit run web_app.py` | 公网链接直接打开 |

## 📄 License

MIT — 自由使用、修改、分发。
