"""本地知识库检索：向量存 SQLite，小 ONNX 模型做嵌入和重排。

刻意不引入 torch：嵌入模型 int8 只有 23 MB，重排 83 MB，
向量检索用 numpy 暴力算余弦，几千条片段以内是毫秒级，
对树莓派这种没有好 GPU 的场景比拉一个向量数据库省事得多。
"""

import re
import sqlite3
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

BASE = Path(__file__).parent
EMBED_DIR = BASE / "models" / "embed" / "bge-small-zh-v1.5"
RERANK_DIR = BASE / "models" / "rerank" / "mxbai-rerank-xsmall-v1"

# bge 系列做检索时给 query 加的前缀，文档侧不加
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

MAX_SEQ_LEN = 512                  # 嵌入模型的位置上限
RAG_TOP_K = 8                      # 向量召回多少条
RAG_RERANK_TOP_N = 3               # 重排之后留几条给模型看
CHUNK_SIZE = 300                   # 切块长度（字符）
CHUNK_OVERLAP = 50                 # 相邻块的重叠，避免把上下文切断


class Embedder:
    """bge-small-zh-v1.5，输出 512 维归一化句向量。"""

    def __init__(self, model_dir: Path = EMBED_DIR):
        model_path = Path(model_dir) / "model_quantized.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"嵌入模型不存在：{model_path}")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        self.dim = self.session.get_outputs()[0].shape[-1]

    def encode(self, texts, is_query: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]

        encodings = [self.tokenizer.encode(t) for t in texts]
        width = max(len(e.ids) for e in encodings)
        input_ids = np.zeros((len(encodings), width), dtype=np.int64)
        attention = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            length = len(encoding.ids)
            input_ids[row, :length] = encoding.ids
            attention[row, :length] = 1

        hidden = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention,
            "token_type_ids": np.zeros_like(input_ids),
        })[0]
        # bge 取 CLS 位当句向量，再归一化，之后点积就等于余弦
        vectors = hidden[:, 0, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.clip(norms, 1e-9, None)).astype(np.float32)


class Reranker:
    """mxbai-rerank-xsmall-v1 交叉编码器，给每个候选打个相关性分。"""

    def __init__(self, model_dir: Path = RERANK_DIR):
        model_path = Path(model_dir) / "model_quantized.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"重排模型不存在：{model_path}")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        encodings = [self.tokenizer.encode(query, doc) for doc in docs]
        width = max(len(e.ids) for e in encodings)
        input_ids = np.zeros((len(encodings), width), dtype=np.int64)
        attention = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            length = len(encoding.ids)
            input_ids[row, :length] = encoding.ids
            attention[row, :length] = 1

        logits = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention,
        })[0]
        # 单标签交叉编码器，过一个 sigmoid 当 0 到 1 的相关性分数
        return (1.0 / (1.0 + np.exp(-logits.reshape(-1)))).tolist()


def chunk_text(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按空行分段后再把短段拼成块，单段超长就硬切并留一点重叠。

    按段拼块而不是一段一块，是为了别把「标题」和它下面的正文拆开——
    一个孤零零的标题被召回出来是没用的。
    """
    chunks = []
    buffer = ""

    def flush():
        nonlocal buffer
        if buffer:
            chunks.append(buffer)
            buffer = ""

    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        if len(paragraph) > size:
            flush()
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start:start + size])
                if start + size >= len(paragraph):
                    break
                start += size - overlap
            continue
        if buffer and len(buffer) + len(paragraph) + 2 > size:
            flush()
        buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph

    flush()
    return chunks


class RagIndex:
    """知识库索引。模型按需加载，没查询就不占内存。"""

    def __init__(self, db_path: Path | str,
                 knowledge_dir: Path | str | None = None):
        self.db_path = Path(db_path)
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else None
        # 检索是在流水线的工作线程里跑的，和创建连接的线程不是同一个，
        # 所以关掉线程检查，自己加锁保证同一时刻只有一个线程用这个连接。
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    source  TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    vector  BLOB NOT NULL
                )
                """
            )
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None
        self._matrix: np.ndarray | None = None
        self._rowids: np.ndarray | None = None

    # ---------- 模型懒加载 ----------
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    # ---------- 建索引 ----------
    def rebuild(self, knowledge_dir: Path | str | None = None) -> int:
        """把知识库目录整个重灌一遍，返回写入的片段数。"""
        source_dir = Path(knowledge_dir or self.knowledge_dir)
        if source_dir is None or not source_dir.is_dir():
            raise FileNotFoundError(f"知识库目录不存在：{source_dir}")

        chunks = []
        for path in sorted(source_dir.rglob("*")):
            if path.suffix.lower() not in (".md", ".txt"):
                continue
            for ordinal, piece in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
                chunks.append((path.name, ordinal, piece))

        with self._lock:
            with self.conn:
                self.conn.execute("DELETE FROM chunks")
                if chunks:
                    vectors = self.embedder.encode([c[2] for c in chunks])
                    self.conn.executemany(
                        "INSERT INTO chunks (source, ordinal, content, vector) "
                        "VALUES (?, ?, ?, ?)",
                        [(s, o, c, v.tobytes()) for (s, o, c), v in zip(chunks, vectors)],
                    )
        self._matrix = None
        self._rowids = None
        return len(chunks)

    def close(self):
        with self._lock:
            self.conn.close()

    # ---------- 查询 ----------
    def _load_matrix(self):
        rows = self.conn.execute("SELECT id, vector FROM chunks ORDER BY id").fetchall()
        if not rows:
            self._rowids = np.zeros(0, dtype=np.int64)
            self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
            return
        self._rowids = np.array([r["id"] for r in rows], dtype=np.int64)
        self._matrix = np.vstack(
            [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
        )

    def search(self, query: str, top_k: int = RAG_TOP_K,
               top_n: int = RAG_RERANK_TOP_N) -> list[dict]:
        """先向量召回 top_k，再用交叉编码器重排，返回前 top_n 条。"""
        with self._lock:
            return self._search(query, top_k, top_n)

    def _search(self, query: str, top_k: int, top_n: int) -> list[dict]:
        if self._matrix is None:
            self._load_matrix()
        if self._rowids.size == 0:
            return []

        query_vector = self.embedder.encode(query, is_query=True)[0]
        scores = self._matrix @ query_vector          # 都归一化过，点积即余弦
        k = min(top_k, scores.size)
        picked = np.argpartition(-scores, k - 1)[:k]
        picked = picked[np.argsort(-scores[picked])]

        ids = [int(self._rowids[i]) for i in picked]
        rows = self.conn.execute(
            f"SELECT id, source, content FROM chunks WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall()
        by_id = {row["id"]: row for row in rows}

        candidates = []
        for index in picked:
            row = by_id.get(int(self._rowids[index]))
            if row is not None:
                candidates.append({
                    "source": row["source"],
                    "content": row["content"],
                    "vector_score": float(scores[index]),
                })

        # 只有一条候选就不值得为它加载重排模型
        if len(candidates) > 1:
            rerank_scores = self.reranker.score(query, [c["content"] for c in candidates])
            for candidate, score in zip(candidates, rerank_scores):
                candidate["rerank_score"] = score
            candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

        return candidates[:top_n]
