"""全本地中文语音助手。

休眠时只跑唤醒词检测，唤醒后做中文识别，
回复交给 OpenAI 兼容接口（知识库和结束对话走工具调用），
回复边生成边合成边播，播放走 sounddevice。

各模块分工：
    config      参数和系统提示词
    audio       麦克风、扬声器
    wakeword    唤醒词检测
    asr         语音识别
    tts         语音合成
    brain       对话、历史、工具
    pipeline    流式播报流水线
    rag         本地知识库检索
"""

import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from asr import SherpaASR, punctuate
from audio import Microphone, Speaker
from brain import ChatBrain, StreamResult
from console import LiveLine, echo_stream
from config import (ASR_MODEL_DIR, DEBUG_TOOLS, HISTORY_DB, KWS_MODEL_DIR,
                    MIC_WARMUP, PAUSE_SEPARATOR, RAG_DB, SILENCE_TIMEOUT,
                    SLEEP_NOTICE, TTS_MAX_SENTENCE_CHARS, TTS_PAUSE_AFTER,
                    TURN_GRACE, WAKE_KEYWORD)
from pipeline import SpeechPipeline
from tts import SherpaTTS
from wakeword import WakeWordDetector
from tools import pi_tools_enabled


class VoiceAssistant:
    """状态机：IDLE 只跑唤醒词，LISTENING 跑识别加对话。"""

    def __init__(self, tts: SherpaTTS, asr: SherpaASR, wakeword: WakeWordDetector,
                 brain: ChatBrain):
        self.mic = Microphone()
        self.speaker = Speaker()
        self.tts = tts
        self.asr = asr
        self.wakeword = wakeword
        self.brain = brain
        self.pipeline = SpeechPipeline(tts, self.speaker, TTS_MAX_SENTENCE_CHARS)

        self.state = "IDLE"
        self.last_voice_time = 0.0
        self._pending: list[str] = []      # 攒着还没交给模型的识别结果
        self._user_line = LiveLine("📝 你说：")

    # ---------- 播报 ----------
    def say(self, text: str):
        """单句播报，走和流式回复同一条流水线。"""
        self._speak(echo_stream("🔊 助手：", iter([text])))

    def _speak(self, text_chunks):
        """边收边合成边播。

        麦克风在播报期间是关着的，播完之后先开流、等它热起来，再响提示音，
        然后丢掉开流和提示音期间采到的帧——这时候才开始算用户的话。
        """
        self.mic.stop()
        self.mic.flush()
        played = False
        try:
            played = self.pipeline.run(text_chunks)
        except Exception as exc:
            print(f"[播报] 出错：{exc}")
        time.sleep(TTS_PAUSE_AFTER)
        # 先把采集流开起来，等它真正开始送帧，再响提示音。
        # 反过来的话用户听到提示音就开口，第一个字会被启动延迟吃掉。
        self.mic.start()
        time.sleep(MIC_WARMUP)
        if played:
            self.speaker.beep(self.tts.sample_rate)
        # 丢掉开流和提示音期间采到的帧，麦克风已经热好了
        self.mic.flush()
        # 播报期间麦克风是停的，静音计时必须从重新开录这一刻算起，
        # 否则生成加播报的时间会被算进静音超时，用户还没开口就回休眠了。
        self.last_voice_time = time.time()

    def run(self):
        self.mic.start()
        print("=" * 50)
        print("🤖 全本地中文语音助手")
        print(f"   唤醒词：{WAKE_KEYWORD}")
        print(f"   静音超时：{SILENCE_TIMEOUT}s")
        print(f"   工具：{'、'.join(self.brain.registry.names())}")
        if not pi_tools_enabled():
            print("   （树莓派工具没注册：当前不是树莓派，")
            print("     想在别的机器上试就把 config.py 里的 PI_TOOLS 设成 True）")
        print("=" * 50)
        print("🎤 休眠中，等待唤醒词...\n")

        try:
            while True:
                data = self.mic.read(timeout=0.5)
                if data is None:
                    continue
                if self.state == "IDLE":
                    self._handle_idle(data)
                else:
                    self._handle_listening(data)
        except KeyboardInterrupt:
            print("\n👋 退出")
        finally:
            self.mic.close()
            self.brain.close()

    # ---------- 休眠状态：只跑唤醒词 ----------
    def _handle_idle(self, data: bytes):
        pcm = np.frombuffer(data, dtype=np.int16)
        if self.wakeword.detect(pcm):
            print("✅ 唤醒")
            self._on_wake()

    # ---------- 监听状态：只跑识别 ----------
    def _handle_listening(self, data: bytes):
        if self.state != "LISTENING":
            return                      # 已经回休眠了，这一帧不该还走这条分支
        text = self.asr.feed(data)
        now = time.time()

        if text:
            # 拿到一段识别结果先攒着，不马上交给模型。
            # 端点检测在连续静音一秒多就触发，用户句中间停一下就会被切断，
            # 而一旦交给模型，麦克风马上就关了，后半句等于对着关掉的麦说的。
            self._pending.append(text)
            self.last_voice_time = now
            self._user_line.update(PAUSE_SEPARATOR.join(self._pending))
        else:
            partial = self.asr.partial()
            if partial:
                # 还在说，端点检测没到，先别让静音超时把对话掐了
                self.last_voice_time = now
                shown = PAUSE_SEPARATOR.join([*self._pending, partial])
                self._user_line.update(shown)

        idle = now - self.last_voice_time
        if self._pending and idle > TURN_GRACE:
            self._commit_turn()
        elif idle > SILENCE_TIMEOUT:
            print("💤 静音超时，回到休眠\n")
            self._on_sleep(announce=True)

    def _commit_turn(self):
        """攒够了，把整段话交给模型。"""
        text = punctuate(self._pending, PAUSE_SEPARATOR)
        self._pending.clear()
        if not text:
            return
        self._user_line.finish(text)
        self._on_user_speech(text)

    # ---------- 状态切换 ----------
    def _on_wake(self):
        self.state = "LISTENING"
        self.last_voice_time = time.time()
        self.asr.reset()
        self.wakeword.reset()
        self._pending.clear()
        self.say("在呢")

    def _on_sleep(self, announce=False):
        """回休眠。announce 为真时先出声说一句，让用户知道助手不听了。"""
        if announce and SLEEP_NOTICE:
            self.say(SLEEP_NOTICE)
        self.state = "IDLE"
        self._pending.clear()
        self.asr.reset()
        self.wakeword.reset()
        self.mic.flush()

    def _on_user_speech(self, text: str):
        result = StreamResult()
        self._speak(echo_stream("🔊 助手：", self.brain.stream_reply(text, result)))
        if DEBUG_TOOLS:
            # 等播报那行收尾了再打，免得和助手说的话挤在同一行
            print(f"🔧 本轮调用了 {result.tools_called} 个工具" if result.tools_called
                  else "🔧 本轮没有调用工具", flush=True)
        if result.sleep:
            print("😴 收到结束意图，直接休眠\n")
            self._on_sleep()


def main():
    base = Path(__file__).parent
    load_dotenv(base / ".env")

    tts = SherpaTTS(base=base)
    asr = SherpaASR(base / ASR_MODEL_DIR)
    wakeword = WakeWordDetector(base / KWS_MODEL_DIR)
    try:
        brain = ChatBrain(
            base_url=os.getenv("OPENAI_BASE_URL", ""),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", ""),
            db_path=base / HISTORY_DB,
            rag_path=base / RAG_DB,
        )
    except RuntimeError as exc:
        raise SystemExit(f"启动失败：{exc}")

    VoiceAssistant(tts, asr, wakeword, brain).run()


if __name__ == "__main__":
    main()
