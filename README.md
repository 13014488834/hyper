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

**Q: 嵌入模型下载失败？**
A: `setup_models.py` 默认走 ModelScope（国内快）。如果失败，手动切换 HuggingFace 镜像：
```bash
# Windows
set HF_ENDPOINT=https://hf-mirror.com
# Mac / Linux
export HF_ENDPOINT=https://hf-mirror.com

python setup_models.py
```

**Q: DeepSeek API 返回 401 或鉴权失败？**
A: 三种可能：
1. Key 格式不对——正确格式是 `sk-` 开头 + 一串字母数字，检查 `.env` 里有没有多余字符
2. Key 已过期——去 https://platform.deepseek.com 重新生成
3. 余额用完——同一页面查看账户余额

**Q: PDF 上传后提示"未找到可提取的文本"？**
A: 你的 PDF 是扫描件（图片格式），pypdf 只能提取文字型 PDF。解决方法：
1. 换用文字型 PDF（Word 导出的那种）
2. 用 OCR 工具（如 Adobe Acrobat）先识别再导入
3. 直接把内容粘贴到 TXT 文件上传

**Q: 回答和知识库内容不相关？**
A: 向量库可能用了旧数据。删除 `chroma_db/` 文件夹再重启，系统会自动用当前知识库重建。本地 CLI 可以直接加 `--rebuild` 参数。

**Q: 启动报 `port 8501 already in use`？**
A: 端口被占用了，换一个：
```bash
streamlit run web_app.py --server.port 8502
```

**Q: Chroma 报错 / 向量库损坏？**
A: 删除 `chroma_db/` 文件夹，重启系统会自动重建。

**Q: Reranker 模型加载失败？**
A: 确保 `python setup_models.py` 完整执行（下载约 1GB）。如果 ModelScope 不可用，脚本会自动回退到 HuggingFace 镜像。

**Q: 云端版和本地版有什么区别？**
A:

| | 本地版 | 云端版 |
|---|---|---|
| 嵌入模型 | 本地 390MB（免费无限次） | HuggingFace API（免费有限流） |
| 混合检索 | ✅ BM25 + 语义 | ✅ BM25 + 语义 |
| Reranker | ✅ 本地 1GB 模型 | ❌ 云端未启用 |
| 部署方式 | `streamlit run web_app.py` | https://aucodhdcwzwwjmucdqhxqz.streamlit.app/ |

**Q: 云端版 HuggingFace API 报 429（请求过多）？**
A: HuggingFace 免费 API 有频率限制。等几分钟再试，或者换成本地版无限制使用。
