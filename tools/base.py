"""工具的通用接口。模型能调什么、怎么调，都由这里的定义决定。"""

from dataclasses import dataclass


@dataclass
class ToolResult:
    """工具执行结果：回给模型的文本，以及是否要顺手结束对话。"""

    text: str
    stop: bool = False


class Tool:
    """一个可以被模型调用的工具。"""

    name = ""
    description = ""
    parameters = {"type": "object", "properties": {}}

    def spec(self):
        """转成 OpenAI 工具调用要的格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments: dict) -> ToolResult:
        raise NotImplementedError

    # ---------- 给子类用的小工具 ----------
    @staticmethod
    def as_int(value, default=None):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


class ToolRegistry:
    """工具集合。模型看到的是 specs，执行走 run。"""

    def __init__(self, tools=()):
        self._tools = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool):
        self._tools[tool.name] = tool
        return tool

    def specs(self):
        return [tool.spec() for tool in self._tools.values()]

    def names(self):
        return list(self._tools)

    def run(self, name: str, arguments: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"没有叫 {name} 的工具。")
        try:
            return tool.run(arguments or {})
        except Exception as exc:
            # 工具出错不该把整轮对话打断，把原因告诉模型让它自己解释
            print(f"[工具] {name} 出错：{exc}")
            return ToolResult(f"{name} 执行失败：{exc}")
