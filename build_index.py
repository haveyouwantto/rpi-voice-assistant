"""把 knowledge/ 目录下的文档切块、向量化，写进 rag.db。

改完知识库文档之后跑一次：
    .\\.venv\\Scripts\\python.exe build_index.py
"""

from pathlib import Path

import rag
from config import KNOWLEDGE_DIR, RAG_DB

BASE = Path(__file__).parent


def main():
    knowledge_dir = BASE / KNOWLEDGE_DIR
    index = rag.RagIndex(BASE / RAG_DB, knowledge_dir)
    try:
        count = index.rebuild(knowledge_dir)
    except FileNotFoundError as exc:
        raise SystemExit(f"建索引失败：{exc}")
    finally:
        index.close()
    print(f"知识库已重建：{count} 个片段 -> {RAG_DB}")


if __name__ == "__main__":
    main()
