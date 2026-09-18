"""中文语音识别。只在监听状态下跑，休眠时不工作。

用 sherpa-onnx 的流式 zipformer2 CTC 模型，中文专用，int8 只有 25 MB。
和唤醒词、语音合成共用同一套 onnxruntime 运行时，不额外引入推理框架。

端点检测交给模型自己做：连续静音就判定一句话说完，不用在外面维护计时器。
"""

import re
from pathlib import Path

import numpy as np
import sherpa_onnx

from config import ASR_MODEL_FILE, ASR_MODEL_DIR, ASR_SAMPLE_RATE, ASR_THREADS


class SherpaASR:
    def __init__(self, model_dir: str | Path = ASR_MODEL_DIR,
                 sample_rate: int = ASR_SAMPLE_RATE, num_threads: int = ASR_THREADS):
        model_dir = Path(model_dir)
        model_path = model_dir / ASR_MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(
                f"识别模型不存在：{model_path}\n"
                f"从 sherpa-onnx 的 asr-models 发布页下载 "
                f"sherpa-onnx-streaming-zipformer-small-ctc-zh-int8，解压到 {model_dir}"
            )

        self.sample_rate = sample_rate
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
            tokens=str(model_dir / "tokens.txt"),
            model=str(model_path),
            num_threads=num_threads,
            sample_rate=sample_rate,
            enable_endpoint_detection=True,
            provider="cpu",
        )
        self.stream = self.recognizer.create_stream()

    def feed(self, pcm_bytes: bytes) -> str | None:
        """喂一帧 int16 采样，说到一句完整的话时返回文本，否则返回 None。"""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.stream.accept_waveform(self.sample_rate, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        if self.recognizer.is_endpoint(self.stream):
            text = self.recognizer.get_result(self.stream)
            self.recognizer.reset(self.stream)
            return text or None
        return None

    def partial(self) -> str:
        """当前这一句的中间结果，用来判断用户还在不在说。"""
        return self.recognizer.get_result(self.stream)

    def reset(self):
        self.recognizer.reset(self.stream)


# 判断疑问语气。两个坑：
#   "吧" 是商量语气（"你走吧"），不算提问；
#   "几个" 前面加"好"就成了数量（"好几个地方"），不能当疑问词。
QUESTION_TAILS = ("吗", "呢", "么")
QUESTION_PATTERN = re.compile(
    r"什么|怎么|为什么|哪|谁|多少|几点|几号|几天|几次|(?<!好)几个|"
    r"是否|是不是|能不能|可不可以|有没有|行不行|要不要|对不对"
)


def looks_like_question(text: str) -> bool:
    stripped = text.rstrip("。？！，、 ")
    if not stripped:
        return False
    if stripped.endswith(QUESTION_TAILS):
        return True
    return bool(QUESTION_PATTERN.search(stripped))


def punctuate(fragments, separator: str = " ") -> str:
    """把几段识别结果拼起来，标出停顿并补上句末标点。

    识别模型是流式 CTC，只看左边上下文，而标点要看完整个句子才能判断，
    所以它不输出任何标点。补的办法：

    - 几段之所以分开，是因为端点检测在中间听到了静音，也就是用户确实停顿了，
      所以每处边界都用 separator 标一下；
    - 结尾按有没有疑问词决定问号还是句号。

    这样送给模型的就不是一长串没头没尾的汉字了。问号尤其重要——
    模型得知道用户是在提问还是在陈述。
    """
    text = separator.join(piece.strip() for piece in fragments if piece.strip())
    if not text:
        return ""
    return text + ("？" if looks_like_question(text) else "。")
