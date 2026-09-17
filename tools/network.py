"""网络状态查询。优先用 ip 命令，拿不到就退回 /proc。"""

import json
import socket

from tools.base import Tool, ToolResult
from tools.pi import read_text, run


def interfaces():
    """返回 [{"name":.., "addrs":[..], "state":..}]。"""
    text = run(["ip", "-j", "addr"])
    if text:
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if data:
            result = []
            for item in data:
                addrs = [info.get("local") for info in item.get("addr_info", [])
                         if info.get("local")]
                result.append({
                    "name": item.get("ifname", "?"),
                    "state": item.get("operstate", "?"),
                    "addrs": addrs,
                })
            return result
    # 退回 hostname -I，只能拿到地址、分不清是哪个网卡
    text = run(["hostname", "-I"])
    if text:
        return [{"name": "未知", "state": "?", "addrs": text.split()}]
    return []


def default_gateway():
    text = read_text("/proc/net/route")
    if not text:
        return None
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 3:
            continue
        if fields[1] == "00000000" and fields[2] != "00000000":
            raw = int(fields[2], 16)
            return ".".join(str((raw >> shift) & 0xFF) for shift in (0, 8, 16, 24))
    return None


def wifi_info():
    """返回 (SSID, 信号质量百分比)。没连无线就是 (None, None)。"""
    ssid = run(["iwgetid", "-r"])
    quality = None
    text = read_text("/proc/net/wireless")
    if text:
        for line in text.splitlines():
            if ":" not in line:
                continue
            fields = line.split(":")
            if len(fields) < 2:
                continue
            numbers = fields[1].split()
            if len(numbers) >= 2:
                try:
                    # 这一列是 70 分制的信号强度
                    quality = round(float(numbers[1].rstrip(".")) / 70 * 100)
                except ValueError:
                    quality = None
                break
    return (ssid or None), quality


def online(host="1.1.1.1", port=53, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class NetworkStatusTool(Tool):
    name = "network_status"
    description = (
        "查这台树莓派的网络状态：网卡和 IP、无线信号、默认网关、能不能连外网。"
        "用户问连上网没有、IP 是多少、WiFi 信号好不好这类问题时调用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "what": {
                "type": "string",
                "enum": ["summary", "interfaces", "wifi", "connectivity"],
                "description": "要看哪一项，不填就给个总览",
            }
        },
    }

    def run(self, arguments):
        what = str(arguments.get("what") or "summary").lower()
        parts = {
            "interfaces": self._interfaces,
            "wifi": self._wifi,
            "connectivity": self._connectivity,
        }
        if what in parts:
            return ToolResult(parts[what]())
        lines = [self._interfaces(), self._wifi(), self._connectivity()]
        return ToolResult("\n".join(line for line in lines if line))

    def _interfaces(self):
        items = interfaces()
        if not items:
            return "读不到网卡信息"
        lines = []
        for item in items:
            if item["name"] == "lo":
                continue
            addrs = "，".join(item["addrs"]) or "没有地址"
            lines.append(f"{item['name']}（{item['state']}）：{addrs}")
        gateway = default_gateway()
        if gateway:
            lines.append(f"默认网关 {gateway}")
        return "\n".join(lines) if lines else "没有可用的网卡"

    def _wifi(self):
        ssid, quality = wifi_info()
        if not ssid:
            return "没有连着无线网络"
        if quality is None:
            return f"无线网络 {ssid}"
        return f"无线网络 {ssid}，信号 {quality}%"

    def _connectivity(self):
        return "能连外网" if online() else "连不上外网"
