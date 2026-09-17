"""模型可以调用的工具集合。

树莓派相关的三个模块：
    system   系统状态，温度、负载、内存、磁盘、供电
    network  网络状态，网卡、IP、无线信号、连通性
    gpio     GPIO 读写，只允许操作 config 里登记过的管脚

非树莓派环境默认不注册这些工具，免得白占上下文；想在别的机器上试就把
config.py 里的 PI_TOOLS 设成 True。
"""

import config
from tools.base import Tool, ToolRegistry, ToolResult
from tools.gpio import GpioListTool, GpioReadTool, GpioWriteTool
from tools.network import NetworkStatusTool
from tools.pi import is_raspberry_pi
from tools.system import SystemStatusTool


def pi_tools_enabled():
    """PI_TOOLS 没写就自动判断，写了就照写的来。"""
    if config.PI_TOOLS is None:
        return is_raspberry_pi()
    return bool(config.PI_TOOLS)


def default_tools():
    """默认要注册的工具。"""
    if not pi_tools_enabled():
        return []
    return [SystemStatusTool(), NetworkStatusTool(),
            GpioListTool(), GpioReadTool(), GpioWriteTool()]


__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "SystemStatusTool",
    "NetworkStatusTool",
    "GpioListTool",
    "GpioReadTool",
    "GpioWriteTool",
    "default_tools",
    "pi_tools_enabled",
]
