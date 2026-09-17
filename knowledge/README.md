# 知识库

这个目录下的 `.md` 和 `.txt` 会被切成片段、向量化，存进 `rag.db`。

改完文档要重新建索引：

```powershell
.\.venv\Scripts\python.exe build_index.py
```

助手不会主动读取这些内容。只有当模型判断当前问题需要查资料时，才会调用
`search_knowledge_base` 工具，检索结果也只在那一次对话里出现，不常驻上下文。

检索分两步：先用 bge-small-zh-v1.5 向量召回 8 条，再用 mxbai-rerank-xsmall-v1
交叉编码器重排，最后只把前 3 条交给模型。

切块按空行分段后，再把短段拼到 300 字左右一块，单段超过 300 字才硬切并留 50 字重叠，
避免把一句话拦腰截断，也避免标题和它的正文被拆到两个块里。
这些参数在 `rag.py` 顶部的 `CHUNK_SIZE` 和 `CHUNK_OVERLAP`。
