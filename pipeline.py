"""把「模型输出 → 拆句 → 合成 → 播放」串成流水线，三个环节各跑各的。

模型吐一句就送去合成，合成好一句就播一句。首句出声只取决于第一句话
什么时候生成完，跟整段回复有多长无关；合成快的引擎则全程不会让播放中断。

两个队列各带一个线程：一个是 LLM 到合成，一个是合成到播放。
队列有上限，所以模型吐太快会自然被合成速度压住，不会把内存撑爆。
"""

import queue
import threading

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
                    pcm = self.tts.synthesize(sentence)
                    if pcm.size:
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
        return played


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
