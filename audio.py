"""麦克风采集和扬声器播放，是唯一直接碰 sounddevice 的地方。"""

import queue
import threading

import numpy as np
import sounddevice as sd

from config import (ASR_SAMPLE_RATE, BLOCK_SIZE, PROMPT_TONE_DURATION,
                    PROMPT_TONE_FREQ, PROMPT_TONE_VOLUME)


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

    def play_stream(self, chunks, sample_rate: int) -> bool:
        """边收边播：chunks 是一个产出 PCM 的可迭代对象，遇到 None 结束。

        用 OutputStream 而不是 sd.play，是因为它能在播放的同时继续往里写，
        合成那边就不用等上一段播完。返回是否真的播出过声音。
        """
        played = False
        stream = None
        with self._lock:
            try:
                for pcm in chunks:
                    if pcm is None or len(pcm) == 0:
                        continue
                    if stream is None:
                        stream = sd.OutputStream(
                            samplerate=sample_rate, channels=1, dtype="int16"
                        )
                        stream.start()
                    stream.write(np.asarray(pcm, dtype=np.int16))
                    played = True
            finally:
                if stream is not None:
                    stream.stop()
                    stream.close()
        return played
