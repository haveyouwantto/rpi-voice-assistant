"""GPIO 读取和控制。

安全上的取舍：模型只能操作 config.py 里登记过的管脚，用名字指代。
不提供按管脚号随意写，也不提供改管脚模式——语音指令驱动的是物理设备，
让模型随便点一根引脚是危险的。

后端按 gpiozero、RPi.GPIO 的顺序找，都没有就明确报告不可用。
设备对象按管脚缓存，避免每次调用都重新初始化、把已有状态冲掉。
"""

import config
from tools.base import Tool, ToolResult

_backend = None
_backend_checked = False


class GpioZeroBackend:
    name = "gpiozero"

    def __init__(self):
        from gpiozero import DigitalInputDevice, DigitalOutputDevice

        self._input_cls = DigitalInputDevice
        self._output_cls = DigitalOutputDevice
        self._inputs = {}
        self._outputs = {}

    def read(self, pin):
        device = self._inputs.get(pin)
        if device is None:
            device = self._input_cls(pin)
            self._inputs[pin] = device
        return int(device.value)

    def write(self, pin, level):
        device = self._outputs.get(pin)
        if device is None:
            # initial_value=None 表示别在创建时就去驱动引脚
            device = self._output_cls(pin, initial_value=None)
            self._outputs[pin] = device
        device.value = level


class RpiGpioBackend:
    name = "RPi.GPIO"

    def __init__(self):
        import RPi.GPIO as GPIO

        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

    def read(self, pin):
        self._gpio.setup(pin, self._gpio.IN)
        return int(self._gpio.input(pin))

    def write(self, pin, level):
        self._gpio.setup(pin, self._gpio.OUT)
        self._gpio.output(pin, level)


def backend():
    """找一个能用的 GPIO 后端，找不到返回 None。"""
    global _backend, _backend_checked
    if not _backend_checked:
        _backend_checked = True
        for factory in (GpioZeroBackend, RpiGpioBackend):
            try:
                _backend = factory()
                print(f"[GPIO] 用上了 {_backend.name}")
                break
            except Exception:
                continue
    return _backend


def resolve_pin(reference):
    """把名字或管脚号换算成 BCM 管脚号。返回 (管脚号, 名字) 或 (None, 原因)。"""
    text = str(reference).strip()
    pins = config.GPIO_PINS
    if text in pins:
        return int(pins[text]), text
    number = None
    try:
        number = int(text)
    except ValueError:
        return None, f"不认识管脚「{reference}」，可用的名字是：{_pin_names()}"
    if number in [int(v) for v in pins.values()]:
        name = next(k for k, v in pins.items() if int(v) == number)
        return number, name
    return None, (f"管脚 {number} 没有登记在 config.py 的 GPIO_PINS 里，"
                  f"已登记的是：{_pin_names()}")


def _pin_names():
    pins = config.GPIO_PINS
    if not pins:
        return "（空，先去 config.py 里加）"
    return "、".join(f"{name}={pin}" for name, pin in pins.items())


def _require():
    """共同的检查：有没有配置、后端在不在。"""
    if not config.GPIO_PINS:
        return None, (f"还没有配置可控管脚。在 config.py 的 GPIO_PINS 里登记，"
                      f"比如 {{\"风扇\": 18}}。")
    device = backend()
    if device is None:
        return None, "这台机器上没有可用的 GPIO 后端（需要装 gpiozero 或 RPi.GPIO）。"
    return device, None


class GpioListTool(Tool):
    name = "gpio_list"
    description = (
        "列出已经登记过的 GPIO 管脚和它们当前的电位。"
        "用户问有哪些管脚、什么设备可以控制时调用。"
    )
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments):
        device, problem = _require()
        if problem:
            return ToolResult(problem)
        lines = []
        for name, pin in config.GPIO_PINS.items():
            try:
                level = device.read(int(pin))
                lines.append(f"{name}（管脚 {pin}）当前{'高' if level else '低'}电位")
            except Exception as exc:
                lines.append(f"{name}（管脚 {pin}）读不到：{exc}")
        return ToolResult("\n".join(lines))


class GpioReadTool(Tool):
    name = "gpio_read"
    description = "读某一个 GPIO 管脚的电位。用户问某个接口现在是通还是断时调用。"
    parameters = {
        "type": "object",
        "properties": {
            "pin": {"type": "string",
                    "description": "管脚名字（config.py 里登记的）或管脚号"},
        },
        "required": ["pin"],
    }

    def run(self, arguments):
        device, problem = _require()
        if problem:
            return ToolResult(problem)
        pin, name = resolve_pin(arguments.get("pin", ""))
        if pin is None:
            return ToolResult(name)
        try:
            level = device.read(pin)
        except Exception as exc:
            return ToolResult(f"管脚 {pin} 读失败：{exc}")
        return ToolResult(f"{name}（管脚 {pin}）现在是{'高' if level else '低'}电位")


class GpioWriteTool(Tool):
    name = "gpio_write"
    description = (
        "把一个登记过的 GPIO 管脚拉高或拉低，用来开关接在上面的设备。"
        "用户让你打开、关闭、通电、断电某样东西时调用。"
        "只能操作 config.py 里登记过的管脚。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pin": {"type": "string",
                    "description": "管脚名字，必须是 config.py 里登记的"},
            "level": {"type": "integer", "enum": [0, 1],
                      "description": "1 拉高（通电、打开），0 拉低（断电、关闭）"},
        },
        "required": ["pin", "level"],
    }

    def run(self, arguments):
        level = self.as_int(arguments.get("level"))
        if level not in (0, 1):
            return ToolResult("level 只能是 0 或 1。")
        device, problem = _require()
        if problem:
            return ToolResult(problem)
        pin, name = resolve_pin(arguments.get("pin", ""))
        if pin is None:
            return ToolResult(name)
        try:
            device.write(pin, level)
        except Exception as exc:
            return ToolResult(f"管脚 {pin} 写入失败：{exc}")
        return ToolResult(f"已经把{name}（管脚 {pin}）拉{'高' if level else '低'}")
