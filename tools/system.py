"""系统状态查询。读的是 /proc 和 /sys，不依赖 psutil。"""

import os
import shutil
import time

from tools.base import Tool, ToolResult
from tools.pi import cpu_count, is_raspberry_pi, load_average, read_int, read_text, run

# vcgencmd get_throttled 返回的位，每一位代表一种供电或降频状态
THROTTLE_BITS = (
    (0, "当前供电不足"),
    (1, "当前 CPU 频率被限制"),
    (2, "当前正在降频"),
    (3, "当前触及软温度上限"),
    (16, "曾经供电不足"),
    (17, "曾经限制过 CPU 频率"),
    (18, "曾经降过频"),
    (19, "曾经触及软温度上限"),
)


def cpu_temperature():
    """摄氏温度。优先读 thermal zone，读不到再问 vcgencmd。"""
    milli = read_int("/sys/class/thermal/thermal_zone0/temp")
    if milli:
        return milli / 1000.0
    text = run(["vcgencmd", "measure_temp"])
    if text and "=" in text:
        try:
            return float(text.split("=")[1].split("'")[0])
        except (ValueError, IndexError):
            return None
    return None


def memory():
    """返回 (总量, 可用量)，单位字节。"""
    text = read_text("/proc/meminfo")
    if not text:
        return None
    values = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key.strip()] = int(parts[0]) * 1024        # /proc 里是 KB
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None:
        return None
    return total, available if available is not None else 0


def uptime_seconds():
    text = read_text("/proc/uptime")
    if not text:
        return None
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return None


def throttle_flags():
    text = run(["vcgencmd", "get_throttled"])
    if not text or "=" not in text:
        return None
    try:
        value = int(text.split("=")[1], 16)
    except (ValueError, IndexError):
        return None
    return [label for bit, label in THROTTLE_BITS if value & (1 << bit)]


def format_duration(seconds):
    if seconds is None:
        return "未知"
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    parts = []
    if days:
        parts.append(f"{days} 天")
    if hours or days:
        parts.append(f"{hours} 小时")
    parts.append(f"{minutes} 分钟")
    return "".join(parts)


def format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("字节", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024


class SystemStatusTool(Tool):
    name = "system_status"
    description = (
        "查这台树莓派的系统状态：温度、CPU 负载、内存、磁盘、运行时间、供电是否正常。"
        "用户问机器热不热、卡不卡、还剩多少空间、开机多久了、有没有欠压这类问题时调用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "what": {
                "type": "string",
                "enum": ["summary", "temperature", "cpu", "memory", "disk",
                         "uptime", "throttle"],
                "description": "要看哪一项，不填就给个总览",
            }
        },
    }

    def run(self, arguments):
        what = str(arguments.get("what") or "summary").lower()
        parts = {
            "temperature": self._temperature,
            "cpu": self._cpu,
            "memory": self._memory,
            "disk": self._disk,
            "uptime": self._uptime,
            "throttle": self._throttle,
        }
        if what in parts:
            text = parts[what]()
            if text:
                return ToolResult(text)
            if not is_raspberry_pi():
                return ToolResult("这台机器不是树莓派，读不到这一项。")
            return ToolResult(f"读不到 {what} 这一项。")
        lines = [text for text in (func() for func in parts.values()) if text]
        if not lines:
            return ToolResult("这台机器上读不到系统信息。")
        return ToolResult("\n".join(lines))

    # ---------- 各项 ----------
    def _temperature(self):
        value = cpu_temperature()
        return f"CPU 温度 {value:.1f} 摄氏度" if value is not None else None

    def _cpu(self):
        lines = []
        average = load_average()
        cores = cpu_count()
        if average:
            lines.append("负载 %.2f / %.2f / %.2f（%d 核）"
                         % (average[0], average[1], average[2], cores))
        idle = read_int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if idle:
            lines.append("当前频率 %.2f GHz" % (idle / 1_000_000))
        return "，".join(lines) if lines else None

    def _memory(self):
        values = memory()
        if not values:
            return None
        total, available = values
        used = total - available
        return "内存共 %s，已用 %s，可用 %s" % (format_bytes(total),
                                               format_bytes(used),
                                               format_bytes(available))

    def _disk(self):
        usage = shutil.disk_usage("/")
        return "根分区共 %s，已用 %s，剩余 %s（%.0f%%）" % (
            format_bytes(usage.total), format_bytes(usage.used),
            format_bytes(usage.free), usage.used * 100 / usage.total)

    def _uptime(self):
        seconds = uptime_seconds()
        if seconds is None:
            return None
        return f"已经运行 {format_duration(seconds)}（自 {time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time() - seconds))} 起）"

    def _throttle(self):
        flags = throttle_flags()
        if flags is None:
            return None
        if not flags:
            return "供电正常，没有发生过欠压或降频"
        return "供电/降频记录：" + "，".join(flags)
