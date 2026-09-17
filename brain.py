"""对话历史落库，以及和 OpenAI 兼容接口的交互（含工具调用）。"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from openai import BadRequestError, OpenAI

import rag
from config import (HISTORY_MAX, HISTORY_STEP, MAX_TOOL_ROUNDS, SYSTEM_PROMPT)


class ChatStore:
    """对话历史落 SQLite，写路径全在事务里，读写在多线程下也安全。"""

    def __init__(self, db_path: str | Path, max_messages=HISTORY_MAX,
                 truncate_step=HISTORY_STEP):
        self.max_messages = max_messages
        self.truncate_step = truncate_step
        # 流水线里读写历史的线程不是创建连接的线程，所以关掉线程检查，
        # 另外自己加锁——sqlite3 只保证单线程使用安全，不保证多线程。
        self.conn = sqlite3.connect(Path(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.row_factory = sqlite3.Row
        # WAL 让读不挡写，synchronous=FULL 保证提交过的事务真的落到盘上
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
                """
            )

    def load(self) -> list[dict]:
        """取最近 max_messages 条，按时间正序返回。"""
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content FROM messages ORDER BY id DESC LIMIT ?
                ) ORDER BY id
                """,
                (self.max_messages,),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append_turn(self, question: str, answer: str):
        """一问一答连同裁剪写在同一个事务里，要么全成要么全不成。"""
        with self._lock:
            with self.conn:
                self.conn.executemany(
                    "INSERT INTO messages (role, content) VALUES (?, ?)",
                    [("user", question), ("assistant", answer)],
                )
                self._truncate()

    def _truncate(self):
        """超过上限时按 truncate_step 为单位丢掉最旧的，避免每轮都删。"""
        total = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        if total <= self.max_messages:
            return
        excess = total - self.max_messages
        drop = ((excess + self.truncate_step - 1) // self.truncate_step) * self.truncate_step
        self.conn.execute(
            """
            DELETE FROM messages WHERE id IN (
                SELECT id FROM messages ORDER BY id LIMIT ?
            )
            """,
            (drop,),
        )

    def close(self):
        with self._lock:
            self.conn.close()


# ==================== 工具 ====================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "查询本地知识库，返回和问题最相关的资料片段。"
                "用户问到需要查资料、查文档，或者你不确定的事实性问题时调用。"
                "打招呼、闲聊、让你做别的事情时不要调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索用的查询词，用用户问题里的关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": (
                "结束对话并让助手进入休眠。"
                "用户说再见、拜拜、退下、没事了、别聊了这类表示要结束的意思时调用。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class StreamResult:
    """流式回复跑完之后才填好：完整文本，以及要不要休眠。"""

    text: str = ""
    sleep: bool = False


class ChatBrain:
    """和 OpenAI 兼容接口对话，上下文交给 ChatStore 落库，知识库走工具调用。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 db_path: str | Path, rag_path: str | Path | None = None,
                 system_prompt=SYSTEM_PROMPT):
        if not api_key:
            raise RuntimeError("没读到 OPENAI_API_KEY，检查项目根目录的 .env 文件。")
        if not model:
            raise RuntimeError("没读到 OPENAI_MODEL，检查项目根目录的 .env 文件。")

        self.client = OpenAI(base_url=base_url or None, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.store = ChatStore(db_path)
        self.rag = rag.RagIndex(rag_path) if rag_path else None
        self.tools_enabled = True

    def close(self):
        self.store.close()
        if self.rag is not None:
            self.rag.close()

    # ---------- 工具执行 ----------
    def _run_tool(self, name: str, raw_arguments: str) -> tuple[str, bool]:
        """执行工具，返回（回给模型的文本，是否要休眠）。"""
        if name == "end_conversation":
            return "已经进入休眠。", True

        if name == "search_knowledge_base":
            if self.rag is None:
                return "本地知识库没有启用。", False
            try:
                arguments = json.loads(raw_arguments or "{}")
            except ValueError:
                return "工具参数解析失败。", False
            query = str(arguments.get("query", "")).strip()
            if not query:
                return "查询词是空的。", False
            try:
                hits = self.rag.search(query)
            except FileNotFoundError as exc:
                return f"知识库用不了：{exc}", False
            if not hits:
                return "知识库里没找到相关内容。", False
            blocks = [f"[{i}] 来自《{hit['source']}》\n{hit['content']}"
                      for i, hit in enumerate(hits, 1)]
            return "\n\n".join(blocks), False

        return f"没有叫 {name} 的工具。", False

    # ---------- 对外：流式 ----------
    def _stream_once(self, messages):
        """流式发一次请求。接口不认 tools 参数时自动退化成普通对话。"""
        kwargs = {"model": self.model, "messages": messages, "stream": True}
        if self.tools_enabled:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"
        try:
            return self.client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if not self.tools_enabled:
                raise
            print(f"[对话] 这个接口不接受工具参数，退化成普通对话：{exc}")
            self.tools_enabled = False
            return self._stream_once(messages)

    def stream_reply(self, text: str, result: StreamResult):
        """逐块产出回复文本，上层可以边收边合成边播。

        跑完之后 result 里是完整文本和是否要休眠，历史也在那时才落库。
        工具调用会中断这一轮流式输出，执行完再发起新一轮。
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.store.load())
        messages.append({"role": "user", "content": text})

        parts: list[str] = []
        should_sleep = False
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                calls: dict[int, dict] = {}
                spoken = ""
                for chunk in self._stream_once(messages):
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    # 工具调用是分片吐的，按 index 攒起来拼完整
                    for partial in (delta.tool_calls or []):
                        slot = calls.setdefault(
                            partial.index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if partial.id:
                            slot["id"] = partial.id
                        if partial.function is not None:
                            if partial.function.name:
                                slot["name"] = partial.function.name
                            if partial.function.arguments:
                                slot["arguments"] += partial.function.arguments
                    if delta.content:
                        spoken += delta.content
                        parts.append(delta.content)
                        yield delta.content

                if not calls:
                    break

                messages.append({
                    "role": "assistant",
                    "content": spoken or None,
                    "tool_calls": [
                        {"id": calls[index]["id"],
                         "type": "function",
                         "function": {"name": calls[index]["name"],
                                      "arguments": calls[index]["arguments"]}}
                        for index in sorted(calls)
                    ],
                })
                for index in sorted(calls):
                    call = calls[index]
                    output, stop = self._run_tool(call["name"], call["arguments"])
                    messages.append({"role": "tool",
                                     "tool_call_id": call["id"],
                                     "content": output})
                    should_sleep = should_sleep or stop

                if should_sleep:
                    break

            if should_sleep and not parts:
                # 模型只调了工具没说话，补一句告别
                parts.append("好的，我先退下了。")
                yield parts[-1]
        except Exception as exc:
            print(f"[对话] 请求失败：{exc}")
            if not parts:
                fallback = "抱歉，我这边网络好像出了问题，你等一下再问我。"
                parts.append(fallback)
                yield fallback
        finally:
            result.text = "".join(parts).strip()
            result.sleep = should_sleep
            if result.text:
                self.store.append_turn(text, result.text)
