"""唤醒词检测。休眠状态下唯一在跑的东西，不做语音识别。

用 sherpa-onnx 的关键词检测模型（zipformer transducer，中英双语训练）。
中文在这个模型里是一等公民：换唤醒词不用重新训练，
把汉字转成模型要的拼音 token 写进 keywords.txt 就行，
转换直接用 sherpa_onnx.text2token，省得手写拼音标调。
"""

from pathlib import Path

import numpy as np
import sherpa_onnx

from config import (ASR_SAMPLE_RATE, KWS_ENCODER, KWS_DECODER, KWS_JOINER,
                    KWS_TOKEN_TYPE, WAKE_KEYWORD, WAKE_KEYWORD_SCORE,
                    WAKE_KEYWORD_THRESHOLD)


class WakeWordDetector:
    def __init__(self, model_dir: str | Path, keyword: str = WAKE_KEYWORD,
                 score: float = WAKE_KEYWORD_SCORE,
                 threshold: float = WAKE_KEYWORD_THRESHOLD):
        model_dir = Path(model_dir)
        tokens = model_dir / "tokens.txt"
        if not tokens.exists():
            raise FileNotFoundError(
                f"唤醒词模型不存在：{tokens}\n"
                f"从 sherpa-onnx 的 kws-models 发布页下载 "
                f"sherpa-onnx-kws-zipformer-zh-en-3M，解压到 {model_dir}"
            )

        self.keyword = keyword
        self.sample_rate = ASR_SAMPLE_RATE
        keywords_file = self._write_keywords(model_dir, tokens, keyword)

        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(tokens),
            encoder=str(model_dir / KWS_ENCODER),
            decoder=str(model_dir / KWS_DECODER),
            joiner=str(model_dir / KWS_JOINER),
            keywords_file=str(keywords_file),
            num_threads=1,
            sample_rate=self.sample_rate,
            keywords_score=score,
            keywords_threshold=threshold,
            provider="cpu",
        )
        self.stream = self.spotter.create_stream()

    @staticmethod
    def _write_keywords(model_dir: Path, tokens: Path, keyword: str) -> Path:
        """把汉字唤醒词转成拼音 token 写成 keywords.txt。

        格式是每行「token 空格分隔 @显示名」，和模型自带的示例一致。
        """
        pieces = sherpa_onnx.text2token(
            [keyword], tokens=str(tokens), tokens_type=KWS_TOKEN_TYPE,
            output_ids=False,
        )[0]
        if not pieces:
            raise ValueError(f"唤醒词转换不出 token：{keyword!r}")
        path = model_dir / "keywords.txt"
        path.write_text(" ".join(pieces) + f" @{keyword}\n", encoding="utf-8")
        return path

    def detect(self, pcm_int16: np.ndarray) -> bool:
        """喂一帧 int16 采样，判断这一帧是否触发唤醒。"""
        samples = pcm_int16.astype(np.float32) / 32768.0
        self.stream.accept_waveform(self.sample_rate, samples)
        while self.spotter.is_ready(self.stream):
            self.spotter.decode_stream(self.stream)
        result = self.spotter.get_result(self.stream)
        if not result:
            return False
        print(f"[唤醒词] 命中 {result}")
        self.spotter.reset_stream(self.stream)
        return True

    def reset(self):
        self.spotter.reset_stream(self.stream)
