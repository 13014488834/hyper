# RAG 问答系统

基于 **LangChain + DeepSeek + Chroma** 的知识库问答系统，支持 **混合检索（BM25 + 语义）+ Reranker 精排**。

**PDF / TXT 知识库 · CLI 命令行 · Web 聊天界面 · Streamlit Cloud 公网部署。**

## 使用方式

### 🌐 在线版（无需安装）

👉 **https://aucodhdcwzwwjmucdqhxqz.streamlit.app/**

浏览器打开即用，上传 PDF/TXT 知识文件，填 DeepSeek API Key，直接问答。

### 💻 本地版（完整功能：混合检索 + Reranker）

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 下载模型（~1.4GB，只需一次）
python setup_models.py

# 3. 配置 Key
#    创建 .env，写入: DEEPSEEK_API_KEY=sk-你的密钥

# 4. 启动 Web
streamlit run web_app.py
# → http://localhost:8501
```

- 📁 上传 PDF/TXT，自动向量化
- 🔀 混合检索（BM25 + 语义，RRF 融合）
- 🎯 Cross-Encoder Reranker 精排
- 🔒 API Key 不存储、不泄露

### ⌨️ CLI 命令行

```bash
python rag_system.py                              # 纯 knowledge.txt
python rag_system.py --pdf contract.pdf            # 带 PDF
python rag_system.py --rebuild                     # 强制重建向量库
python rag_system.py --no-hybrid --no-reranker     # 关闭混合检索/Reranker
```

## 项目结构

```
├── rag_core.py          # 核心管道（嵌入、向量库、LLM、混合检索、Reranker）
├── pdf_loader.py        # PDF/TXT 文件加载
├── rag_system.py        # CLI 命令行入口
├── web_app.py           # 本地 Web 界面（完整功能）
├── web_app_cloud.py     # 云端 Web 界面（Streamlit Cloud）
├── setup_models.py      # 模型下载脚本（嵌入 + Reranker）
├── knowledge.txt        # 默认知识库
├── requirements.txt     # 依赖
├── .env                 # API Key（不上传 Git）
```

## 技术栈

| 组件 | 用途 |
|------|------|
| LangChain | RAG 管道编排 |
| DeepSeek API | 大模型生成答案 |
| Chroma | 向量数据库（语义检索） |
| BM25 | 关键词检索（混合检索） |
| RRF | Reciprocal Rank Fusion（双路融合） |
| Cross-Encoder | Reranker 精排 |
| Streamlit | Web 界面 |
| pypdf | PDF 文本提取 |

## 常见问题

**Q: 给 HR / 面试官看，怎么用？**
A: 发公网链接 https://aucodhdcwzwwjmucdqhxqz.streamlit.app/ ，对方直接打开即可。

**Q: 我的 API Key 会泄露吗？**
A: 不会。`.env` 在 `.gitignore` 中，Web 界面每个用户自己填 Key，不存储。
