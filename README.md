# 全本地中文语音助手

休眠时只跑唤醒词检测，唤醒后做中文识别，
回复交给 OpenAI 兼容接口，最后本地合成、sounddevice 播放。

唤醒、识别、合成都跑在本地，只有对话内容会发给你在 `.env` 里配置的模型接口。

语音相关的模型都是 ONNX，靠 onnxruntime 推理，整套不依赖 PyTorch。

## 目录结构

```
main.py                                  入口：装配各部件，跑状态机
config.py                                所有参数和系统提示词
audio.py                                 麦克风采集、扬声器播放
wakeword.py                              唤醒词检测（sherpa-onnx KWS）
asr.py                                   语音识别（Vosk）
tts.py                                   语音合成（sherpa-onnx Kokoro）
brain.py                                 对话、历史落库、工具调用
pipeline.py                              流式播报流水线
rag.py                                   知识库检索（嵌入、重排、向量存取）
build_index.py                           重建知识库索引
requirements.txt                         依赖清单
.env                                     对话模型配置（base_url / key / model）
chat_history.db                          对话历史库，自动生成
rag.db                                   知识库向量，build_index.py 生成
knowledge/                               知识库源文档，放 .md 和 .txt
models/
  asr/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/   中文识别（25 MB）
  tts/vits-icefall-zh-aishell3/          中文合成，默认（29 MB，RTF 约 27）
  tts/kokoro-int8-multi-lang-v1_1/       中文合成，备选（140 MB，Apache-2.0）
  kws/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/   唤醒词模型（中英，31 MB）
  embed/bge-small-zh-v1.5/               中文嵌入模型（int8 ONNX，23 MB）
  rerank/mxbai-rerank-xsmall-v1/         多语重排模型（int8 ONNX，83 MB）
```

`models/vosk-model-small-cn-0.22.zip` 是解压前的原始压缩包，可以删掉。
`models/piper/` 和 `models/wakewords/` 是旧方案留下的，现在没用了，确认没问题也可以删。

## 安装

依赖已经装在项目内的 `.venv` 里，不需要再动系统 Python。要重新装一遍的话：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 配置对话模型

编辑项目根目录的 `.env`，三个值都要填：

```
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=你的密钥
OPENAI_MODEL=gpt-4o-mini
```

接口格式按 OpenAI 的标准写，换成任何兼容服务只要改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。
key 或者 model 留空会在启动时直接报错，不会静默失败。

对话历史存在 `chat_history.db`（SQLite），重启后接着上次聊。入库的三条约束：

- 一问一答连同裁剪写在同一个事务里，要么全成要么全不成，中途出错整体回滚。
- `journal_mode = WAL` 加 `synchronous = FULL`，提交过的事务真正落盘，掉电不会丢也不会坏。
- 消息内容全程走参数化查询，语音转出来的文本里带 SQL 也只会当成普通字符串存。

上下文上限 100 条，超了按 50 条为单位丢掉最旧的，这两个值分别是
`config.py` 里的 `HISTORY_MAX` 和 `HISTORY_STEP`。
接口请求失败时助手会口头提示出错，并且不会把这次失败的问答写进历史。

表结构在 `brain.py` 的 `ChatStore` 里，就一张 `messages` 表，
带 `role`、`content`、`created_at`，直接拿 sqlite3 命令或 DB Browser 都能看。

## 语音识别

用 sherpa-onnx 的流式 zipformer2 CTC 模型，中文专用，int8 只有 25 MB，
每帧 80 毫秒喂进去，靠模型自带的端点检测（连续静音）判断一句话说完了。
和唤醒词、合成共用同一套 onnxruntime，不额外引入运行时。

## 语音合成

用 icefall 的 Matcha 中文模型，跑在 onnxruntime 上，22050 Hz。

选它是因为它是非自回归加显式时长建模，架构上属于 FastSpeech 那一支：
解码过程没有随机采样，同一句话反复合成输出的样本数在 63993 到 64210 之间
（约千分之三的波动），不会出现韵律每次都变的情况。实测 RTF 约 6，
走流式播报首句 0.55 秒出声。用 ASR 回读的相似度是 0.85、0.91、0.78。

对比参考：VITS 那一类自带随机时长预测器，韵律靠采样，
同一句话的样本数会抖到 ±5%，而且能找到的中文 VITS 模型输出只有 8 kHz。

模型需要两个文件：声学模型 `model-steps-3.onnx`，以及从 `vocoder-models`
发布页下的声码器 `hifigan_v2.onnx`（3.6 MB），都放在 `models/tts/` 下。

`TTS_SPEED` 调语速。**授权要注意**：训练数据 Baker 通常只授权非商用。

## 流式播报

`pipeline.py` 把「模型输出 → 拆句 → 合成 → 播放」串成流水线，三个环节各跑各的：

```
模型逐字吐字 --句子--> 队列 --合成--> 队列 --播放--> 扬声器
```

模型吐出一句就送去合成，合成好一句就播一句，不用等整段回复生成完。
两个队列都有上限，模型吐太快会被合成速度自然压住，不会撑爆内存。

拆句优先在句末标点断，一直等不到就在逗号处断，
单句最长 `TTS_MAX_SENTENCE_CHARS` 个字符（默认 40）。

## 本地知识库

把 `.md` 或 `.txt` 丢进 `knowledge/`，然后重建索引：

```powershell
.\.venv\Scripts\python.exe build_index.py
```

检索**不常驻**。知识库内容不进系统提示词，只有模型判断当前问题需要查资料时，
才会调用 `search_knowledge_base` 工具，结果也只在那一次对话里出现。

检索分两步：先用 `bge-small-zh-v1.5` 向量召回 8 条，再用 `mxbai-rerank-xsmall-v1`
交叉编码器重排，只把前 3 条交给模型。两个模型都是 int8 ONNX，按需加载，
不查询就不占内存，也不需要装 PyTorch。

向量存在 `rag.db` 的 `chunks` 表里，检索时用 numpy 暴力算余弦。
几千个片段以内是毫秒级，不值得为此引入向量数据库。

## 工具调用

模型有两个工具，都是按需触发，不占常驻上下文：

- `search_knowledge_base`，查上面的本地知识库。
- `end_conversation`，用户说再见、退下、没事了这类结束语时调用，
  助手说完告别语就直接回休眠，不再多花一轮请求。

如果用的接口不支持 `tools` 参数，助手会自动去掉工具参数重试一次，
之后按普通对话跑，不会因此罢工。

## 运行

```powershell
.\.venv\Scripts\python.exe main.py
```

启动后先说唤醒词（默认「小白小白」），听到"在呢"就可以说中文了；
安静超过 `SILENCE_TIMEOUT` 秒（当前 20 秒）自动回到休眠。

## 换唤醒词

改 [config.py](config.py) 里的 `WAKE_KEYWORD` 就行，**不需要重新训练**。
汉字会由 `sherpa_onnx.text2token` 转成模型要的拼音 token 写进模型的 `keywords.txt`：

```
小白小白  ->  x iǎo b ái x iǎo b ái @小白小白
```

调不准的时候动这两个值：

- `WAKE_KEYWORD_SCORE`，关键词增强分，越大越容易命中（默认 1.5）。
- `WAKE_KEYWORD_THRESHOLD`，触发阈值，越大越难触发（默认 0.25）。

误唤醒多就调低 score 或调高 threshold，喊了没反应就反过来。

用的是 zipformer transducer 关键词检测模型，中英双语训练，跑 chunk-8 那组
（比 chunk-16 延迟低），每帧 80 毫秒，和录音分帧一致。

## 说明

- 改系统提示词找 `config.py` 里的 `SYSTEM_PROMPT`，换对话逻辑看 `brain.py` 的 `ChatBrain`。
- 录音格式固定 16 kHz / 单声道 / int16，每帧 1280 采样（80 ms），
  唤醒和识别都按这个粒度喂。
- 唤醒词模型只认真人语音。用 TTS 合成的声音去测是测不出来的，
  这类模型是在真人录音上训的，合成语音的频谱特征和它对不上。
- 换 TTS 或识别引擎时注意模型目录：路径都写在 `config.py` 里，
  模型本身从 sherpa-onnx 的 `tts-models` / `asr-models` / `kws-models`
  发布页下载解压，不用改代码。
