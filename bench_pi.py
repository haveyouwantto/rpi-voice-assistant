"""在树莓派上跑一遍，看这套东西能不能实时。

用法（在项目根目录）：
    .venv/bin/python bench_pi.py

会依次量唤醒词、识别、合成三段的耗时，检查音频设备，最后给结论和调参建议。
缺哪个模型就跳过哪一段，不会因为少一个文件整个跑不起来。
"""

import os
import sys
import time
import wave

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

BAR = "=" * 62


def section(title):
    print(f"\n{BAR}\n{title}\n{BAR}")


def verdict(ok, warn=None):
    if ok:
        return "够用"
    return warn or "不够"


# ==================== 机器信息 ====================
def show_machine():
    section("机器信息")
    try:
        from tools.pi import cpu_count, is_raspberry_pi, load_average, model_name
        from tools.system import cpu_temperature, memory, format_bytes, format_duration, uptime_seconds

        name = model_name() or "读不到型号"
        print(f"  型号      {name}")
        print(f"  是树莓派  {is_raspberry_pi()}")
        print(f"  CPU 核数  {cpu_count()}")
        temp = cpu_temperature()
        print(f"  CPU 温度  {temp:.1f} 度" if temp else "  CPU 温度  读不到")
        values = memory()
        if values:
            print(f"  内存      {format_bytes(values[0])}（可用 {format_bytes(values[1])}）")
        up = uptime_seconds()
        if up:
            print(f"  已运行    {format_duration(up)}")
        average = load_average()
        if average:
            print(f"  负载      {average[0]:.2f} / {average[1]:.2f} / {average[2]:.2f}")
    except Exception as exc:
        print(f"  读机器信息出错：{exc}")

    print(f"  Python    {sys.version.split()[0]}")
    try:
        import onnxruntime
        print(f"  推理引擎  onnxruntime {onnxruntime.__version__}")
    except Exception:
        pass
    try:
        import sherpa_onnx
        print(f"  sherpa    {sherpa_onnx.__version__}")
    except Exception:
        pass


# ==================== 音频设备 ====================
def show_audio():
    section("音频设备")
    try:
        import sounddevice as sd

        print("  默认输入  %s" % (sd.query_devices(sd.default.device[0])["name"],))
        print("  默认输出  %s" % (sd.query_devices(sd.default.device[1])["name"],))

        for rate in (8000, 22050, 24000, 44100, 48000):
            try:
                sd.check_output_settings(samplerate=rate, channels=1, dtype="int16")
                state = "支持"
            except Exception as exc:
                state = f"不支持（{str(exc)[:40]}）"
            print(f"  输出 {rate:>6} Hz  {state}")
    except Exception as exc:
        print(f"  音频设备检查出错：{exc}")


# ==================== 唤醒词 ====================
def bench_wakeword():
    section("唤醒词（每帧 80 毫秒，必须跑得比 80 毫秒快）")
    try:
        from config import KWS_MODEL_DIR, WAKE_KEYWORD
        from wakeword import WakeWordDetector

        started = time.monotonic()
        detector = WakeWordDetector(os.path.join(BASE, KWS_MODEL_DIR))
        print(f"  加载耗时  {time.monotonic() - started:.2f} 秒")
        print(f"  唤醒词    {WAKE_KEYWORD}")

        frame = np.zeros(1280, dtype=np.int16)
        for _ in range(10):                       # 预热
            detector.detect(frame)

        rounds = 200
        started = time.monotonic()
        for _ in range(rounds):
            detector.detect(frame)
        per_frame = (time.monotonic() - started) / rounds
        budget = 0.080
        print(f"  每帧耗时  {per_frame * 1000:.2f} 毫秒（预算 80 毫秒）")
        print(f"  余量      {budget / per_frame:.1f} 倍")
        print(f"  结论      CPU 占用约 {per_frame / budget * 100:.1f}% 的一个核")
        return per_frame, budget
    except Exception as exc:
        print(f"  跳过：{exc}")
        return None


# ==================== 识别 ====================
def pick_audio():
    """找一段语音来测识别：优先用模型自带的测试音频。"""
    from config import ASR_MODEL_DIR

    folder = os.path.join(BASE, ASR_MODEL_DIR, "test_wavs")
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.endswith(".wav"):
                path = os.path.join(folder, name)
                with wave.open(path, "rb") as handle:
                    rate = handle.getframerate()
                    data = np.frombuffer(handle.readframes(handle.getnframes()),
                                         dtype=np.int16)
                return data, rate, f"模型自带 {name}"

    try:                                          # 退而求其次：自己合成一段
        from tts import SherpaTTS
        engine = SherpaTTS()
        pcm = engine.synthesize("现在室内温度二十三度，湿度百分之六十。")
        return pcm, engine.sample_rate, "自己合成的语音"
    except Exception:
        return None, None, None


def bench_asr():
    section("语音识别（必须比实时快）")
    try:
        from config import ASR_MODEL_DIR
        from asr import SherpaASR
    except Exception as exc:
        print(f"  跳过：{exc}")
        return None

    try:
        started = time.monotonic()
        asr = SherpaASR(os.path.join(BASE, ASR_MODEL_DIR))
        print(f"  加载耗时  {time.monotonic() - started:.2f} 秒")

        data, rate, source = pick_audio()
        if data is None:
            print("  找不到可用的测试音频，跳过")
            return None
        print(f"  测试音频  {source}，{len(data) / rate:.2f} 秒")

        if rate != 16000:                         # 模型要 16k，先转一下
            count = int(len(data) * 16000 / rate)
            data = np.interp(np.linspace(0, len(data), count, endpoint=False),
                             np.arange(len(data)), data).astype(np.int16)

        asr.reset()
        started = time.monotonic()
        for start in range(0, len(data) - 1280 + 1, 1280):
            asr.feed(data[start:start + 1280].tobytes())
        cost = time.monotonic() - started
        audio = len(data) / 16000
        print(f"  解码耗时  {cost:.2f} 秒（音频 {audio:.2f} 秒）")
        print(f"  实时率    {audio / cost:.1f} 倍")
        return audio / cost
    except Exception as exc:
        print(f"  跳过：{exc}")
        return None


# ==================== 合成 ====================
SENTENCES = [
    "现在室内温度二十三度，湿度百分之六十。",
    "外面下着小雨，建议带上伞再出门。",
    "另外你下午三点有个会议，别忘了。",
]


def bench_tts_engine(engine):
    from tts import SherpaTTS

    started = time.monotonic()
    tts = SherpaTTS(engine=engine)
    load = time.monotonic() - started
    tts.synthesize("预热")

    audio = cost = 0.0
    for sentence in SENTENCES:
        began = time.monotonic()
        pcm = tts.synthesize(sentence)
        cost += time.monotonic() - began
        audio += pcm.size / tts.sample_rate
    return tts.sample_rate, audio, cost, load


def bench_tts():
    section("语音合成（实时率大于 1 才能边合成边播）")
    results = {}
    for engine in ("matcha", "vits"):
        try:
            rate, audio, cost, load = bench_tts_engine(engine)
        except Exception as exc:
            print(f"  {engine:<7} 跳过：{str(exc)[:60]}")
            continue
        rtf = audio / cost
        results[engine] = (rate, rtf)
        print(f"  {engine:<7} 采样率 {rate:>5}  加载 {load:>5.1f}s  "
              f"音频 {audio:>5.1f}s  合成 {cost:>5.1f}s  实时率 {rtf:>6.1f}")
    return results


# ==================== 结论 ====================
def summarize(wake, asr_rtf, tts_results):
    section("结论")
    from config import TTS_ENGINE, TTS_PREBUFFER

    if wake:
        per_frame, budget = wake
        if per_frame < budget:
            print(f"  唤醒词  没问题，每帧只用了预算的 {per_frame / budget * 100:.0f}%")
        else:
            print(f"  唤醒词  跟不上，每帧 {per_frame * 1000:.0f} 毫秒超过 80 毫秒预算")

    if asr_rtf:
        print("  识别    " + ("没问题" if asr_rtf > 1.5 else "有点紧，可能会漏字")
              + f"，实时率 {asr_rtf:.1f}")

    if not tts_results:
        print("  合成    没有可用的模型")
        return

    print()
    for engine, (rate, rtf) in tts_results.items():
        marks = []
        if engine == TTS_ENGINE:
            marks.append("当前在用")
        if engine == max(tts_results, key=lambda key: tts_results[key][1]):
            marks.append("最快")
        suffix = f"  <- {'、'.join(marks)}" if marks else ""
        print(f"  合成 {engine:<7} 采样率 {rate:>5}  实时率 {rtf:>6.1f}{suffix}")

    print()
    # 先看当前在用的够不够，够就别折腾
    current = tts_results.get(TTS_ENGINE)
    if current and current[1] >= 1.5:
        print(f"  => 当前用的 {TTS_ENGINE} 有 {current[1]:.1f} 倍实时，够用，不用动。")
        return
    if current and current[1] >= 1.0:
        print(f"  => {TTS_ENGINE} 只有 {current[1]:.2f} 倍实时，勉强跟得上。"
              f"把 TTS_PREBUFFER 调到 2 会更稳（现在是 {TTS_PREBUFFER}）。")
        return

    fastest = max(tts_results.items(), key=lambda item: item[1][1])
    engine, (rate, rtf) = fastest
    head = f"  当前用的 {TTS_ENGINE} " + (
        f"只有 {current[1]:.2f} 倍实时" if current else "没有可用模型") + "。"
    if rtf >= 1.5:
        print(head + f"换成 {engine} 有 {rtf:.1f} 倍实时，够用。")
        print(f'     把 config.py 里的 TTS_ENGINE 改成 "{engine}"，'
              f"代价是采样率从 {tts_results[TTS_ENGINE][0] if current else '?'} "
              f"降到 {rate}。")
    elif rtf >= 1.0:
        print(head + f"最快的 {engine} 也只有 {rtf:.2f} 倍，"
              f"把 TTS_PREBUFFER 调到 60（现在 {TTS_PREBUFFER}）会更稳。")
    else:
        print(head + f"最快的 {engine} 也只有 {rtf:.2f} 倍实时，"
              f"合成比播放慢 {1 / rtf:.1f} 倍，边合成边播一定会断续。")
        print(f"     TTS_PREBUFFER 调到 60 以上（现在 {TTS_PREBUFFER}），"
              f"等于整段合成完再播。")
        print("     如果声音还很糊，那就只能换更快的板子了。")


def main():
    print(BAR)
    print("树莓派性能自检")
    print(BAR)
    show_machine()
    show_audio()
    wake = bench_wakeword()
    asr_rtf = bench_asr()
    tts_results = bench_tts()
    summarize(wake, asr_rtf, tts_results)
    print()


if __name__ == "__main__":
    main()
