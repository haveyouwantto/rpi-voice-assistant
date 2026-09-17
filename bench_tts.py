"""在目标机器上量一下语音合成到底跑多快。

桌面和树莓派的差距能有十倍，光看我给的数字没用，得在真机上跑：
    .venv/bin/python bench_tts.py

实时率（RTF）= 音频时长 / 合成耗时。这个值的含义：
    > 1.5   流式播报很顺，边合成边播完全跟得上
    1 ~ 1.5 勉强跟得上，prebuffer 调大一点更稳
    < 1     合成比实时慢，播到后面一定断续，得开大 prebuffer 或者换小模型
"""

import time

from tts import SherpaTTS

SENTENCES = [
    "现在室内温度二十三度，湿度百分之六十。",
    "外面下着小雨，建议带上伞再出门。",
    "另外你下午三点有个会议，别忘了。",
]


def main():
    tts = SherpaTTS()
    print(f"采样率 {tts.sample_rate}，线程数见 config.TTS_THREADS")
    tts.synthesize("预热")

    total_audio = total_cost = 0.0
    print(f"{'文本':<22}{'音频':>8}{'耗时':>8}{'RTF':>8}")
    for sentence in SENTENCES:
        started = time.monotonic()
        pcm = tts.synthesize(sentence)
        cost = time.monotonic() - started
        seconds = pcm.size / tts.sample_rate
        total_audio += seconds
        total_cost += cost
        print(f"{sentence:<22}{seconds:>7.2f}s{cost:>7.2f}s{seconds / cost:>8.2f}")

    rtf = total_audio / total_cost
    print(f"{'合计':<22}{total_audio:>7.2f}s{total_cost:>7.2f}s{rtf:>8.2f}")

    print()
    if rtf > 1.5:
        print("结论：跟得上实时，流式播报没问题。")
    elif rtf > 1.0:
        print("结论：勉强跟得上。TTS_PREBUFFER 调到 2 会更稳。")
    else:
        print("结论：合成比实时慢。两个选择——")
        print("  1. 把 config.TTS_PREBUFFER 调大（比如 60），"
              "等于整段合成完再播，代价是开口变慢")
        print("  2. 换更小的模型：vits-piper-zh_CN-xiao_ya-medium-int8 只有 13 MB，"
              "比现在的 matcha 小五倍")


if __name__ == "__main__":
    main()
