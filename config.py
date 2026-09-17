"""所有可调参数和系统提示词集中在这里，其它模块只读不写。"""

# ==================== 音频 ====================
ASR_SAMPLE_RATE = 16000
BLOCK_SIZE = 1280                  # 80ms 一帧，唤醒和识别都按这个粒度喂
SILENCE_TIMEOUT = 20.0             # 唤醒后多久没说话就回休眠
TTS_PAUSE_AFTER = 0.15             # 播报结束后多等一会儿再开麦克风

# 播报结束后提示"可以说了"的短音
PROMPT_TONE_FREQ = 880             # Hz，调高更尖，调低更闷
PROMPT_TONE_DURATION = 0.09        # 秒，改成 0 就不播提示音
PROMPT_TONE_VOLUME = 0.25          # 0 到 1

# ==================== 唤醒词 ====================
KWS_MODEL_DIR = "models/kws/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
WAKE_KEYWORD = "小白小白"           # 改这里就能换唤醒词，不用重新训练
KWS_TOKEN_TYPE = "ppinyin"         # 汉字转拼音 token，模型要的形式
WAKE_KEYWORD_SCORE = 1.5           # 关键词增强分，越大越容易命中
WAKE_KEYWORD_THRESHOLD = 0.25      # 触发阈值，越大越难触发

# 用 chunk-8 的那组，比 chunk-16 延迟更低
KWS_ENCODER = "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"
KWS_DECODER = "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"
KWS_JOINER = "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"

# ==================== 对话 ====================
HISTORY_DB = "chat_history.db"     # 对话历史数据库，相对项目根目录
HISTORY_MAX = 100                  # 上下文最多保留多少条（不含 system）
HISTORY_STEP = 50                  # 超过上限时，一次丢掉最旧的多少条
MAX_TOOL_ROUNDS = 4                # 一次对话里最多来回几轮工具调用

# ==================== 知识库 ====================
RAG_DB = "rag.db"
KNOWLEDGE_DIR = "knowledge"

# ==================== 语音识别 ====================
ASR_MODEL_DIR = "models/asr/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
ASR_MODEL_FILE = "model.int8.onnx"
ASR_THREADS = 2

# ==================== 语音合成 ====================
# 非自回归加显式时长建模，架构上属于 FastSpeech 那一支，输出确定不飘。
# 训练数据 Baker 通常限非商用。
TTS_MODEL_DIR = "models/tts/matcha-icefall-zh-baker"
TTS_VOCODER = "models/tts/hifigan_v2.onnx"
TTS_SPEAKER_ID = 0                 # 单说话人模型，只能填 0
TTS_SPEED = 1.0                    # 大于 1 更快
TTS_THREADS = 4

# 流式播报：模型一边吐字一边攒句子，攒够这些字就送去合成
TTS_MAX_SENTENCE_CHARS = 40


SYSTEM_PROMPT = """你是一个运行在树莓派上的智能语音助手，叫做小白，负责通过麦克风与用户进行纯语音交流，并通过扬声器播放你的回答。

为了保证 TTS（语音合成）朗读的自然与流畅，你必须严格遵守以下规则：

1. 语言风格口语化：使用自然、亲切、通俗易懂的口语表达，像朋友聊天一样回题，避免过于冷冰冰的书面语和晦涩的学术长句。
2. 禁止油腻的表达方式：严格禁止使用“不是……而是……”等矫情的句式；严格禁止使用“炸裂”、“扎心”、“震撼”、“绝绝子”等浮夸的词汇。
3. 绝对禁止视觉格式化：回答中严禁出现任何 Markdown 标记（如 # 标题、**加粗**、- 列表、| 表格 |、代码块、URL 链接等）。所有内容必须是纯文本，直接可读。
4. 直奔主题，拒绝废话套话：开头直接回答问题。绝对禁止使用“先说结论”、“不绕弯子直接说”、“总而言之”等欲盖弥彰的废话开场白。
5. 优化朗读听感与标点：仅使用逗号、句号、问号和感叹号。避免使用括号、破折号、省略号或特殊符号，确保 TTS 引擎能够平滑停顿。
6. 规范数字与英文读音：遇到数字、单位时，尽量转换为适合口语朗读的形式（例如：将 "8KB" 写为 "8 KB" 或 "8千字节"，将 "100m" 写为 "100米"，将 "3.14" 写为 "三点一四"），防止语音引擎误读。
7. 控制篇幅，直奔主题：语音交互中听者的注意力有限，回答要精炼直接，一针见血。长话短说，单次回答尽量控制在三到五句话以内，重要内容优先表达。
8. 适合听觉理解：避免列举长串的数据或复杂的代数公式，遇到复杂概念时，用生活中的比喻快速进行口头解释。
9. 猜测提问意图：语音输入可能不准确，出现大量同音字词，遇到不理解和不通顺的地方尝试猜测提问者的意图，重点猜测读音相近的字词，如果进行了猜测，那么需要在回答开头加上“你想说的是不是……”。
"""
