"""终端输出小工具：流式内容顺着打，不要刷一屏。"""

import sys


class LiveLine:
    """先在原地反复刷新一行，定型时才换行。

    识别的中间结果是一帧一帧来的，直接 print 会刷出几十行。
    所以用回车回到行首覆盖，再按上一行的宽度补空格把残留擦掉。

    输出被重定向到文件时没有光标可回，原地刷新的回车只会把日志搅乱，
    这种情况就只在定型时打一次。
    """

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.enabled = sys.stdout.isatty()
        self._width = 0

    def update(self, text: str):
        """刷新当前行，不换行。"""
        if not self.enabled:
            return
        self._draw(self.prefix + text, newline=False)

    def finish(self, text: str = ""):
        """定型并换行。"""
        if not self.enabled:
            print(self.prefix + text, flush=True)
            return
        self._draw(self.prefix + text, newline=True)

    def _draw(self, line: str, newline: bool):
        padding = " " * max(0, self._width - len(line))
        print("\r" + line + padding, end="\n" if newline else "", flush=True)
        self._width = 0 if newline else len(line)


def echo_stream(prefix: str, chunks):
    """把流式文本边收边打到同一行，同时原样往下传。

    pipeline 需要的是完整的 chunk 序列，终端要的是即时可见，
    所以在这里串一下：打一行前缀，之后来多少打多少，结束才换行。

    前缀要等第一个字到了再打。模型可能先调工具，这时候还没有正文，
    提前把前缀打出来会让那一行空挂着，工具的输出没处落脚。
    """
    started = False
    try:
        for chunk in chunks:
            if not started:
                print(prefix, end="", flush=True)
                started = True
            print(chunk, end="", flush=True)
            yield chunk
    finally:
        # 消费者提前退出也要把行收干净，不能把光标留在半截
        if started:
            print(flush=True)
