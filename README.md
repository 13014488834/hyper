# RAG 问答系统

基于 **LangChain + DeepSeek + Chroma + HuggingFace** 的本地知识库问答系统。

## 功能

- 读取 `knowledge.txt` 作为知识库，自动切分、向量化、持久化
- 命令行交互问答，检索 Top-3 片段，显示答案和引用来源
- 国内网络友好：嵌入模型通过 ModelScope 下载，不走 HuggingFace
- 支持 DeepSeek API（从 `.env` 读取 key）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env` 文件，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-你的密钥
```

（去 [platform.deepseek.com](https://platform.deepseek.com) 创建）

### 3. 下载嵌入模型

```bash
python setup_models.py
```

模型会下载到 `./models/` 目录（约 390MB），只需执行一次。

### 4. 准备知识库

编辑 `knowledge.txt`，放入你的知识内容（支持中文）。

### 5. 启动

```bash
python rag_system.py
```

启动后输入问题即可问答，输入 `quit` 退出。

## 项目结构

```
├── rag_system.py        # 主程序
├── setup_models.py      # 模型下载脚本
├── knowledge.txt        # 知识库文件
├── .env                 # API Key 配置（不提交到 Git）
├── requirements.txt     # Python 依赖
├── models/              # 本地嵌入模型（不提交到 Git）
└── chroma_db/           # 向量库持久化（不提交到 Git）
```

## 配置参数

在 `rag_system.py` 顶部可修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | 500 | 文档切分大小 |
| `CHUNK_OVERLAP` | 50 | 片段重叠长度 |
| `TOP_K` | 3 | 检索返回数量 |
| `LLM_TEMPERATURE` | 0.1 | 生成随机性 |
