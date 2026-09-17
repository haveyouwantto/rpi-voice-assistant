"""麦克风采集和扬声器播放，是唯一直接碰 sounddevice 的地方。"""

import queue
import sys
import threading

import numpy as np
import sounddevice as sd

from config import (ASR_SAMPLE_RATE, AUDIO_LATENCY, BLOCK_SIZE,
                    PROMPT_TONE_DURATION, PROMPT_TONE_FREQ, PROMPT_TONE_VOLUME,
                    SILENCE_ALSA_ERRORS, TTS_PREBUFFER)

_alsa_handler = None          # 必须留个引用，回调被回收会让进程直接崩


def silence_alsa_errors():
    """把 ALSA 的错误处理器换成空函数。

    underrun 这类提示是 ALSA 在 C 层直接往 stderr 打的，Python 的日志和
    重定向都拦不住，只能用 ctypes 换掉它的处理函数。只在 Linux 上有效。
    """
    global _alsa_handler
    if not sys.platform.startswith("linux"):
        return False
    try:
        from ctypes import CFUNCTYPE, c_char_p, c_int, cdll

        handler_type = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
        _alsa_handler = handler_type(lambda *args: None)
        cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(_alsa_handler)
        return True
    except Exception:
        return False


class Microphone:
    """把采集回调丢进队列，主循环按帧取。"""

    def __init__(self, sample_rate=ASR_SAMPLE_RATE, block_size=BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=block_size,
            callback=self._on_audio,
        )

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            print(f"[麦克风] {status}")
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def start(self):
        if not self._stream.active:
            self._stream.start()

    def stop(self):
        if self._stream.active:
            self._stream.stop()

    def close(self):
        self._stream.close()

    def read(self, timeout=0.5):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class Speaker:
    def __init__(self):
        self._lock = threading.Lock()
        if SILENCE_ALSA_ERRORS:
            silence_alsa_errors()

    def play(self, pcm: np.ndarray, sample_rate: int):
        if pcm.size == 0:
            return
        with self._lock:
            sd.play(pcm, samplerate=sample_rate, blocking=True)

    def beep(self, sample_rate: int, freq=PROMPT_TONE_FREQ,
             duration=PROMPT_TONE_DURATION, volume=PROMPT_TONE_VOLUME):
        """播一个短促的提示音，告诉用户轮到他说了。"""
        if duration <= 0:
            return
        count = int(sample_rate * duration)
        if count <= 0:
            return
        t = np.linspace(0.0, duration, count, endpoint=False)
        wave = np.sin(2 * np.pi * freq * t)
        # 首尾各留 15% 做淡入淡出，不然会有咔哒声
        fade = max(1, count * 15 // 100)
        wave[:fade] *= np.linspace(0.0, 1.0, fade)
        wave[-fade:] *= np.linspace(1.0, 0.0, fade)
        self.play((wave * volume * 32767).astype(np.int16), sample_rate)

    def stop(self):
        sd.stop()

    def play_stream(self, chunks, sample_rate: int,
                    prebuffer: float = TTS_PREBUFFER) -> bool:
        """边收边播，开播前先攒够 prebuffer 秒。

        chunks 产出 PCM，遇到 None 结束，返回是否真的播出过声音。

        合成比实时慢的机器（树莓派就是）一有声音就开播的话，声卡很快供不上，
        于是 ALSA 一直报 underrun、声音断续。先攒一段再开播，小幅卡顿就被
        吸收掉了；prebuffer 设得足够大就等于"整段合成完再播"。
        """
        target = int(max(prebuffer, 0.0) * sample_rate)
        pending: list = []
        buffered = 0
        stream = None
        with self._lock:
            try:
                for pcm in chunks:
                    if pcm is None or len(pcm) == 0:
                        continue
                    if stream is None:
                        pending.append(np.asarray(pcm, dtype=np.int16))
                        buffered += len(pcm)
                        if buffered < target:
                            continue
                        stream = self._open_stream(sample_rate)
                        for piece in pending:
                            stream.write(piece)
                        pending = []
                        continue
                    stream.write(np.asarray(pcm, dtype=np.int16))
                if stream is None and pending:
                    # 还没攒够就结束了，剩下的一起播出去
                    stream = self._open_stream(sample_rate)
                    for piece in pending:
                        stream.write(piece)
            finally:
                if stream is not None:
                    stream.stop()
                    stream.close()
        return stream is not None

    @staticmethod
    def _open_stream(sample_rate: int):
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16",
                                 latency=AUDIO_LATENCY)
        stream.start()
        return stream
