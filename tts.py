"""中文语音合成，只负责吐出 int16 PCM，播放交给 audio.Speaker。

用 sherpa-onnx 跑 icefall 的 Matcha 中文模型。Matcha 是非自回归加显式时长建模，
架构上属于 FastSpeech 那一支：解码过程没有随机采样，同一句话反复合成结果一致，
不会出现韵律每次都变的情况。声学模型加一个 HiFiGAN 声码器，实测 RTF 约 6。

中文的分词、声调和数字读法由模型自带的 lexicon 加规则 FST 处理，不依赖 espeak。
"""

from pathlib import Path

import numpy as np
import sherpa_onnx

from config import (TTS_MODEL_DIR, TTS_SPEAKER_ID, TTS_SPEED, TTS_THREADS,
                    TTS_VOCODER)

# 句末标点，流式播报时按这些断句
SENTENCE_END = "。！？!?；;…\n"
# 实在等不到句末标点，就在这些地方断
CLAUSE_END = "，,、：:"


def split_sentences(chunks, max_chars: int):
    """把流式文本切成一句一句。

    遇到句末标点就断；一直不来句末标点又攒够 max_chars 字，
    就退而求其次在逗号处断，免得第一句话迟迟送不出去。
    """
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        while buffer:
            cut = _find_cut(buffer, max_chars)
            if cut is None:
                break
            piece = buffer[:cut].strip()
            buffer = buffer[cut:]
            if piece:
                yield piece
    if buffer.strip():
        yield buffer.strip()


def _find_cut(buffer: str, max_chars: int) -> int | None:
    for index, char in enumerate(buffer):
        if char in SENTENCE_END:
            return index + 1
    if len(buffer) < max_chars:
        return None
    for index in range(max_chars - 1, max_chars // 2, -1):
        if buffer[index] in CLAUSE_END:
            return index + 1
    return max_chars


class SherpaTTS:
    """对外只有 synthesize 和 sample_rate，主流程不用关心模型细节。"""

    def __init__(self, model_dir: str | Path = TTS_MODEL_DIR,
                 vocoder: str | Path = TTS_VOCODER,
                 speaker_id: int = TTS_SPEAKER_ID, speed: float = TTS_SPEED,
                 num_threads: int = TTS_THREADS, base: Path | None = None):
        base = Path(base) if base else Path(__file__).parent
        model_dir = base / model_dir
        vocoder = base / vocoder

        acoustic = model_dir / "model-steps-3.onnx"
        if not acoustic.exists():
            raise FileNotFoundError(
                f"合成模型不存在：{acoustic}\n"
                f"从 sherpa-onnx 的 tts-models 发布页下载 matcha-icefall-zh-baker"
            )
        if not vocoder.exists():
            raise FileNotFoundError(
                f"声码器不存在：{vocoder}\n"
                f"从 sherpa-onnx 的 vocoder-models 发布页下载 hifigan_v2.onnx"
            )

        self.tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=str(acoustic),
                    vocoder=str(vocoder),
                    lexicon=str(model_dir / "lexicon.txt"),
                    tokens=str(model_dir / "tokens.txt"),
                ),
                num_threads=num_threads,
                provider="cpu",
            ),
            rule_fsts=",".join(
                str(model_dir / name)
                for name in ("phone.fst", "date.fst", "number.fst")
                if (model_dir / name).exists()
            ),
            max_num_sentences=1,
        ))
        self.sample_rate = self.tts.sample_rate
        self.speaker_id = speaker_id
        self.speed = speed

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.array([], dtype=np.int16)
        audio = self.tts.generate(text, sid=self.speaker_id, speed=self.speed)
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size == 0:
            return np.array([], dtype=np.int16)
        return np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
