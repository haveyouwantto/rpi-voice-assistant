"""把「模型输出 → 拆句 → 合成 → 播放」串成流水线，三个环节各跑各的。

模型吐一句就送去合成，合成好一句就播一句。首句出声只取决于第一句话
什么时候生成完，跟整段回复有多长无关；合成快的引擎则全程不会让播放中断。

两个队列各带一个线程：一个是 LLM 到合成，一个是合成到播放。
队列有上限，所以模型吐太快会自然被合成速度压住，不会把内存撑爆。
"""

import queue
import threading
import time

from tts import split_sentences


class SpeechPipeline:
    def __init__(self, tts, speaker, max_sentence_chars: int):
        self.tts = tts
        self.speaker = speaker
        self.max_sentence_chars = max_sentence_chars

    def run(self, text_chunks) -> bool:
        """text_chunks 产出文本片段，返回是否真的播出过声音。"""
        sentences: queue.Queue = queue.Queue(maxsize=8)
        audios: queue.Queue = queue.Queue(maxsize=32)
        errors: list[BaseException] = []
        stats = {"audio": 0.0, "cost": 0.0}

        def produce():
            try:
                for sentence in split_sentences(text_chunks, self.max_sentence_chars):
                    sentences.put(sentence)
            except BaseException as exc:      # noqa: BLE001 - 要带回主线程
                errors.append(exc)
            finally:
                sentences.put(None)

        def synthesize():
            try:
                while True:
                    sentence = sentences.get()
                    if sentence is None:
                        break
                    started = time.monotonic()
                    pcm = self.tts.synthesize(sentence)
                    stats["cost"] += time.monotonic() - started
                    if pcm.size:
                        stats["audio"] += pcm.size / self.tts.sample_rate
                        audios.put(pcm)
            except BaseException as exc:      # noqa: BLE001
                errors.append(exc)
            finally:
                audios.put(None)

        producer = threading.Thread(target=produce, daemon=True)
        worker = threading.Thread(target=synthesize, daemon=True)
        producer.start()
        worker.start()

        try:
            played = self.speaker.play_stream(_drain(audios), self.tts.sample_rate)
        finally:
            # 播放中途出错时把队列腾空，否则合成线程会卡在 put 上等不到人取
            _unblock(audios)
            producer.join(timeout=5)
            worker.join(timeout=5)

        if errors:
            raise errors[0]
        self._report_speed(stats)
        return played

    @staticmethod
    def _report_speed(stats):
        """合成跟不上就提醒，并算出该攒多少缓冲，省得对着断续的声音猜。

        设实时率为 r、这段音频总长 D、开播前攒了 P 秒。播放时缓冲区是
        P + r·t - t，要全程不为负就得 P >= D·(1-r)。r 越接近 1 需要的越少，
        r 到 1 以上就完全不需要攒——所以"一律调成 60"是错的做法，白等。
        """
        if stats["cost"] < 1.0 or stats["audio"] <= 0:
            return
        ratio = stats["audio"] / stats["cost"]
        if ratio >= 1.3:
            return
        audio = stats["audio"]
        need = max(3.0, audio * (1 - min(ratio, 1.0)) * 1.5)
        print(f"⚠️ 合成 {ratio:.2f} 倍实时（{audio:.1f}s 音频花了 "
              f"{stats['cost']:.1f}s），留给播放的余量很薄。", flush=True)
        print(f"   把 config.TTS_PREBUFFER 设成 {need:.0f} 左右，"
              f"这轮就不会断。", flush=True)
        if ratio < 0.7:
            print("   合成比播放慢不少，开口前要等一会儿；"
                  "换更快的模型或者让回复短一点会好很多。", flush=True)


def _drain(items):
    while True:
        item = items.get()
        if item is None:
            return
        yield item


def _unblock(items):
    while True:
        try:
            items.get_nowait()
        except queue.Empty:
            return
