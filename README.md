# RAG 问答系统

基于 **LangChain + DeepSeek + Chroma + HuggingFace** 的本地知识库问答系统。

**支持 PDF / TXT 知识库 + CLI 命令行 + Web 聊天界面。**

## 快速开始（3 步）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载嵌入模型（约 390MB，只需一次）
python setup_models.py

# 3. 配置 DeepSeek API Key
#    创建 .env 文件，写入：DEEPSEEK_API_KEY=sk-你的密钥
#    去 https://platform.deepseek.com 免费获取
```

## 使用方式

### Web 界面（推荐）

```bash
streamlit run web_app.py
# 浏览器打开 http://localhost:8501
```

- 📁 上传 PDF / TXT 文件，自动向量化
- 💬 聊天式问答，带引用来源
- ⚙️ 可调 Temperature、Top-K、Chunk Size
- 🔒 每个用户填自己的 API Key，不会泄露

### CLI 命令行

```bash
python rag_system.py                              # 纯 knowledge.txt
python rag_system.py --pdf contract.pdf            # 带 PDF
python rag_system.py --pdf a.pdf b.pdf             # 多个 PDF
python rag_system.py --txt extra.txt               # 额外 TXT
python rag_system.py --rebuild                     # 强制重建向量库
```

## 项目结构

```
├── rag_core.py          # 核心管道（嵌入、向量库、LLM、RAG 链）
├── pdf_loader.py        # PDF/TXT 文件加载
├── rag_system.py        # CLI 命令行入口
├── web_app.py           # Streamlit Web 界面
├── setup_models.py      # 嵌入模型下载脚本
├── knowledge.txt        # 默认知识库（可替换）
├── requirements.txt     # Python 依赖
├── .env                 # API Key（不上传 Git，需自行创建）
├── models/              # 本地嵌入模型（不上传 Git，setup_models.py 生成）
└── chroma_db/           # 向量库（不上传 Git，自动生成）
```

## 依赖项

| 组件 | 用途 |
|------|------|
| LangChain | RAG 管道编排 |
| DeepSeek API | 大模型生成答案 |
| Chroma | 向量数据库 |
| HuggingFace / ModelScope | 嵌入模型下载与推理 |
| Streamlit | Web 界面 |
| pypdf | PDF 文本提取 |

## 常见问题

**Q: 嵌入模型下载失败？**
A: `setup_models.py` 默认使用 ModelScope（国内快）。海外用户可设环境变量 `HF_ENDPOINT=""` 走 HuggingFace 官方。

**Q: Web 模式给别人用，我的 API Key 会泄露吗？**
A: 不会。Web 界面默认不读取 `.env`，每个用户需自己填写 Key，Key 只存在浏览器会话中，关闭即消失。`.env` 文件在 `.gitignore` 中，不会上传到 GitHub。
