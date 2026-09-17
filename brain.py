"""对话历史落库，以及和 OpenAI 兼容接口的交互（含工具调用）。"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from openai import BadRequestError, OpenAI

import rag
from config import (DEBUG_TOOLS, HISTORY_HOP, HISTORY_MAX, MAX_TOOL_ROUNDS,
                    SYSTEM_PROMPT)
from tools import Tool, ToolRegistry, ToolResult, default_tools


def brief(text, limit=120):
    """把要打印的内容压成一行，太长就截断。"""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


class ChatStore:
    """对话历史落 SQLite，写路径全在事务里，读写在多线程下也安全。"""

    def __init__(self, db_path: str | Path, window=HISTORY_MAX, hop=HISTORY_HOP):
        self.window = window
        self.hop = hop
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

    def load(self, window=None, hop=None) -> list[dict]:
        """取送进模型的那段上下文，按时间正序。

        不是"最近 N 条"那种滑窗——那种每多一句就整体挪一格，前缀一直在变，
        服务端缓存用不上，行为也跟着抖。这里把窗口起点对齐到 hop 的整数倍：
        每攒够 hop 条才整体前移一次，中间那些轮次上下文一字不变。

        库里的数据一条不删，窗口只决定这一次带哪些进上下文。
        """
        window = self.window if window is None else window
        hop = self.hop if hop is None else hop
        with self._lock:
            total = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            if total <= window:
                offset, count = 0, total
            else:
                # 往上取整到 hop 的倍数，保证最后一条永远在窗口里
                offset = ((total - window + hop - 1) // hop) * hop
                count = window
            rows = self.conn.execute(
                "SELECT role, content FROM messages ORDER BY id LIMIT ? OFFSET ?",
                (count, offset),
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

    def close(self):
        with self._lock:
            self.conn.close()

    # ---------- 查过去 ----------
    def _turn_around(self, message_id):
        """把某条消息所属的那一问一答取出来。调用方要持锁。"""
        row = self.conn.execute(
            "SELECT id, role, content, created_at FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            return None
        if row["role"] == "user":
            question = row
            answer = self.conn.execute(
                "SELECT content FROM messages WHERE id > ? AND role = 'assistant' "
                "ORDER BY id LIMIT 1",
                (row["id"],),
            ).fetchone()
        else:
            answer = row
            question = self.conn.execute(
                "SELECT id, content, created_at FROM messages "
                "WHERE id < ? AND role = 'user' ORDER BY id DESC LIMIT 1",
                (row["id"],),
            ).fetchone()
        source = question or row
        return {
            "asked_at": source["created_at"],
            "question": source["content"],
            "answer": answer["content"] if answer else "",
        }

    def search(self, keyword=None, limit=10, days=None):
        """找过去的对话。不给关键词就是最近聊过的几条。"""
        sql = ["SELECT id FROM messages"]
        where, params = [], []
        if keyword:
            where.append("content LIKE ?")
            params.append(f"%{keyword}%")
        if days:
            where.append("created_at >= datetime('now', 'localtime', ?)")
            params.append(f"-{int(days)} days")
        if where:
            sql.append("WHERE " + " AND ".join(where))
        sql.append("ORDER BY id DESC LIMIT ?")
        params.append(limit)

        with self._lock:
            ids = [row["id"] for row in self.conn.execute(" ".join(sql), params)]
            turns, seen = [], set()
            for message_id in ids:
                turn = self._turn_around(message_id)
                # 一问一答常常都被关键词命中，按问题去重
                if turn and turn["question"] not in seen:
                    seen.add(turn["question"])
                    turns.append(turn)
        return turns


# ==================== 应用级工具 ====================
class KnowledgeBaseTool(Tool):
    """查本地知识库。检索结果只回给模型，不进常驻上下文。"""

    name = "search_knowledge_base"
    description = (
        "查询本地知识库，返回和问题最相关的资料片段。"
        "用户问到需要查资料、查文档，或者你不确定的事实性问题时调用。"
        "打招呼、闲聊、让你做别的事情时不要调用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "检索用的查询词，用用户问题里的关键词"},
        },
        "required": ["query"],
    }

    def __init__(self, index):
        self.index = index

    def run(self, arguments):
        if self.index is None:
            return ToolResult("本地知识库没有启用。")
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolResult("查询词是空的。")
        try:
            hits = self.index.search(query)
        except FileNotFoundError as exc:
            return ToolResult(f"知识库用不了：{exc}")
        if not hits:
            return ToolResult("知识库里没找到相关内容。")
        blocks = [f"[{i}] 来自《{hit['source']}》\n{hit['content']}"
                  for i, hit in enumerate(hits, 1)]
        return ToolResult("\n\n".join(blocks))


class EndConversationTool(Tool):
    name = "end_conversation"
    description = (
        "结束对话并让助手进入休眠。"
        "用户说再见、拜拜、退下、没事了、别聊了这类表示要结束的意思时调用。"
    )
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments):
        return ToolResult("已经进入休眠。", stop=True)


class SearchHistoryTool(Tool):
    """翻过去的对话。这就是长期记忆——答案来自数据库，不是模型凭印象编的。"""

    name = "search_history"
    description = (
        "翻过去的对话记录。用户说「上次」「之前」「我跟你说过」「还记得吗」"
        "这类需要回忆的话时调用。也可以不给关键词，看看最近都聊过什么。"
        "更早的对话不在你的上下文里，必须调这个工具去翻，不要凭印象猜。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "要回忆的关键词；不填就返回最近聊过的几条",
            },
            "days": {
                "type": "integer",
                "description": "只看最近几天的记录；不填就是全部历史",
            },
        },
    }

    def __init__(self, store, limit=8):
        self.store = store
        self.limit = limit

    def run(self, arguments):
        keyword = str(arguments.get("keyword", "")).strip() or None
        days = self.as_int(arguments.get("days"))
        turns = self.store.search(keyword=keyword, limit=self.limit, days=days)
        if not turns:
            if keyword:
                return ToolResult(f"翻过了，以前没有聊过「{keyword}」。")
            return ToolResult("还没有更早的对话记录。")
        lines = []
        for turn in turns:
            stamp = turn["asked_at"]
            lines.append(f"[{stamp}] 你说：{brief(turn['question'], 80)}")
            if turn["answer"]:
                lines.append(f"[{stamp}] 我答：{brief(turn['answer'], 80)}")
        return ToolResult("\n".join(lines))


@dataclass
class StreamResult:
    """流式回复跑完之后才填好：完整文本，以及要不要休眠。"""

    text: str = ""
    sleep: bool = False
    tools_called: int = 0


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
        self.registry = ToolRegistry(
            [KnowledgeBaseTool(self.rag), SearchHistoryTool(self.store),
             EndConversationTool(), *default_tools()])
        self.tools_enabled = True

    def close(self):
        self.store.close()
        if self.rag is not None:
            self.rag.close()

    # ---------- 对外：流式 ----------
    def _stream_once(self, messages):
        """流式发一次请求。接口不认 tools 参数时自动退化成普通对话。"""
        kwargs = {"model": self.model, "messages": messages, "stream": True}
        if self.tools_enabled:
            kwargs["tools"] = self.registry.specs()
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
        calls_made = 0
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
                    try:
                        arguments = json.loads(call["arguments"] or "{}")
                    except ValueError:
                        arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    if DEBUG_TOOLS:
                        print(f"🔧 调用 {call['name']} {brief(arguments)}", flush=True)
                    outcome = self.registry.run(call["name"], arguments)
                    calls_made += 1
                    if DEBUG_TOOLS:
                        print(f"🔧 返回 {brief(outcome.text)}", flush=True)
                    messages.append({"role": "tool",
                                     "tool_call_id": call["id"],
                                     "content": outcome.text})
                    should_sleep = should_sleep or outcome.stop

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
            result.tools_called = calls_made
            if result.text:
                self.store.append_turn(text, result.text)
