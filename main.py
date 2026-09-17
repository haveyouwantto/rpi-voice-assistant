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

from asr import SherpaASR
from audio import Microphone, Speaker
from brain import ChatBrain, StreamResult
from console import LiveLine, echo_stream
from config import (ASR_MODEL_DIR, HISTORY_DB, KWS_MODEL_DIR, RAG_DB,
                    SILENCE_TIMEOUT, TTS_MAX_SENTENCE_CHARS, TTS_PAUSE_AFTER,
                    WAKE_KEYWORD)
from pipeline import SpeechPipeline
from tts import SherpaTTS
from wakeword import WakeWordDetector


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
        self._user_line = LiveLine("📝 你说：")

    # ---------- 播报 ----------
    def say(self, text: str):
        """单句播报，走和流式回复同一条流水线。"""
        self._speak(echo_stream("🔊 助手：", iter([text])))

    def _speak(self, text_chunks):
        """边收边合成边播。期间关掉麦克风，播完响一声提示音再开。"""
        self.mic.stop()
        self.mic.flush()
        played = False
        try:
            played = self.pipeline.run(text_chunks)
        except Exception as exc:
            print(f"[播报] 出错：{exc}")
        time.sleep(TTS_PAUSE_AFTER)
        self.mic.flush()
        if played:
            self.speaker.beep(self.tts.sample_rate)
        self.mic.start()
        # 播报期间麦克风是停的，静音计时必须从重新开录这一刻算起，
        # 否则生成加播报的时间会被算进静音超时，用户还没开口就回休眠了。
        self.last_voice_time = time.time()

    def run(self):
        self.mic.start()
        print("=" * 50)
        print("🤖 全本地中文语音助手")
        print(f"   唤醒词：{WAKE_KEYWORD}")
        print(f"   静音超时：{SILENCE_TIMEOUT}s")
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
        text = self.asr.feed(data)

        if text:
            self._user_line.finish(text)
            self.last_voice_time = time.time()
            self._on_user_speech(text)
        else:
            partial = self.asr.partial()
            if partial:
                # 还在说，端点检测没到，先别让静音超时把对话掐了
                self.last_voice_time = time.time()
                self._user_line.update(partial)

        if time.time() - self.last_voice_time > SILENCE_TIMEOUT:
            print("💤 静音超时，回到休眠\n")
            self._on_sleep()

    # ---------- 状态切换 ----------
    def _on_wake(self):
        self.state = "LISTENING"
        self.last_voice_time = time.time()
        self.asr.reset()
        self.wakeword.reset()
        self.say("在呢")

    def _on_sleep(self):
        self.state = "IDLE"
        self.asr.reset()
        self.wakeword.reset()
        self.mic.flush()

    def _on_user_speech(self, text: str):
        result = StreamResult()
        self._speak(echo_stream("🔊 助手：", self.brain.stream_reply(text, result)))
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
