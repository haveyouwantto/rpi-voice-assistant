"""树莓派底层读取。

所有函数在非树莓派环境（比如开发用的 Windows）都返回 None 或空列表，
不抛异常，方便上层统一用"读不到"来兜底。
"""

import os
import shutil
import subprocess

MODEL_FILES = ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model")


def read_text(path, limit=8192):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit).strip("\x00").strip()
    except OSError:
        return None


def read_int(path):
    text = read_text(path)
    if text is None:
        return None
    try:
        return int(text.split()[0])
    except (ValueError, IndexError):
        return None


def run(command, timeout=5):
    """跑一条命令取标准输出。命令不存在、超时、非零退出都返回 None。"""
    if not shutil.which(command[0]):
        return None
    try:
        finished = subprocess.run(command, capture_output=True, text=True,
                                  timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.strip()


def model_name():
    for path in MODEL_FILES:
        text = read_text(path)
        if text:
            return text
    return None


def is_raspberry_pi():
    name = model_name()
    return bool(name and "raspberry pi" in name.lower())


def cpu_count():
    return os.cpu_count() or 1


def load_average():
    """1/5/15 分钟平均负载。Windows 上没有这个概念，返回 None。"""
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None
