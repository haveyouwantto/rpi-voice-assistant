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
tools/                                   树莓派工具（系统、网络、GPIO）
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

### 上下文是 hop 窗口，不是滑窗

每次带进模型的是一段 100 条的消息，但**窗口起点按 25 条对齐**：
每攒够 25 条消息窗口才整体前移一次，中间那些轮次取到的内容一字不变。

滑窗（永远取最近 100 条）每多一句就整体挪一格，前缀一直在变，
服务端缓存复用不上，模型看到的开头也跟着抖。对齐之后稳定得多。

代价是窗口宽度会在 76 到 100 之间锯齿状变化：刚前移时老消息整批掉出去，
但**最后一条永远在窗口里**，不会漏掉刚说的话。

```
轮数  库里  起点  取到
 50   100     0   100
 51   102    25    77
 75   150    50   100
 76   152    75    77
```

窗口宽度和步长是 `config.py` 里的 `HISTORY_MAX` 和 `HISTORY_HOP`。

### 数据库全量保留

一条都不删。上下文窗口只决定这次带哪些进模型，跟存什么无关。
老对话留在库里给 `search_history` 翻，这才是长期记忆的来源。
实测写 130 轮（260 条）后全部还在，最老的一条可以正常搜到。

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

## 在树莓派上跑

把项目目录和 `models/` 拷过去，装好依赖，先跑一次自检：

```bash
.venv/bin/python bench_pi.py
```

它会依次量机器信息、音频设备、唤醒词每帧耗时、识别实时率，以及两个合成引擎的
实时率，最后给结论。缺哪个模型就跳过哪一段，不会因为少一个文件整个跑不起来。

输出长这样（这是桌面上的参考值，树莓派上数字会小很多）：

```
唤醒词  每帧耗时 3.28 毫秒（预算 80 毫秒）
识别    解码耗时 0.19 秒（音频 5.61 秒）  实时率 29.8 倍
合成 matcha  采样率 22050  实时率   5.8  <- 当前在用
合成 vits    采样率  8000  实时率  26.7  <- 最快
=> 当前用的 matcha 有 5.8 倍实时，够用，不用动。
```

三个判断标准：唤醒词每帧要快于 80 毫秒；识别实时率要大于 1；
合成实时率大于 1.5 才能舒服地边合成边播。

只想单独量合成的话还有 `bench_tts.py`。

### 合成跟不上会怎样

RTF 小于 1 时合成比播放慢，声卡供不上就会：

- 终端刷 `ALSA lib pcm.c:8777:(snd_pcm_recover) underrun occurred`
- 声音断断续续

这不是音频配置问题，是合成太慢。两个办法：

| 办法 | 做法 | 代价 |
|---|---|---|
| 攒够再播 | `TTS_PREBUFFER` 调大，比如 60（等于整段合成完再播） | 开口变慢 |
| 换快模型 | `TTS_ENGINE` 改成 `vits` | 音质下降 |

`TTS_PREBUFFER` 默认 0.8 秒，指开播前先攒够多少秒音频。
攒一段能把小幅卡顿吸收掉；设得比整段回复还长，效果就是"合成完再播"。

助手跑完一轮会在终端自查一句，比如：

```
⚠️ 合成只有 0.50 倍实时（3.0s 音频花了 6.0s），播报会断续。
```

看到这个就直接调 `TTS_PREBUFFER`。

### 两个引擎怎么选

| 引擎 | 体积 | 采样率 | 桌面 RTF | 回读相似度 |
|---|---|---|---|---|
| `matcha`（默认） | 75.7 MB | 22050 | 7.1 | 0.79 |
| `vits` | 29.1 MB | 8000 | 30.7 | 0.58 |

回读相似度是把合成结果丢给 ASR 转回文字、再和原文比出来的，越高越清楚。

**速度的差距主要来自采样率**，不是模型大小。vits 快四倍是因为它只生成
8000 个采样点每秒，vocoder 的活儿比 22050 少了将近三倍。所以"找个小模型"
这个方向在这里是走不通的——换成 13 MB 的 int8 音色反而更慢，
因为 x86 上 int8 要反复反量化，而它还是 22 kHz。

同样，把 matcha 自己量化成 int8 也没用：体积从 72 MB 降到 36 MB，
但 RTF 从 5.8 掉到 3.7。动态量化只作用于矩阵乘，matcha 真正吃时间的
卷积部分动不了。

结论：**要实时就得接受 8 kHz**。树莓派上先跑 `bench_tts.py` 量一下，
matcha 如果能有 1 倍实时就不用换；差得远就把 `TTS_ENGINE` 改成 `vits`。

### 其它几个开关

- `TTS_THREADS` 默认是 `CPU 核数 - 1`，特意留一个核给声卡。全占满必 underrun。
- `AUDIO_LATENCY`（0.4 秒）是声卡缓冲区，调大更能扛住偶发卡顿。
- `SILENCE_ALSA_ERRORS` 默认开：ALSA 的 underrun 提示是它在 C 层直接往
  stderr 打的，Python 拦不住，只能用 ctypes 把它的错误处理器换掉。
  它只是提示，ALSA 自己会恢复，不想要这行刷屏就让它闭着嘴。

ASR 是 int8 的 25 MB 模型，树莓派上通常不是瓶颈。

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

工具都是按需触发，不占常驻上下文。应用级的三个：

- `search_knowledge_base`，查上面的本地知识库。
- `search_history`，翻过去的对话记录。用户说「上次」「之前」「我跟你说过」这类话时调用，
  可以给关键词，也可以不给（那就是最近聊过什么），还能按天数限定范围。
- `end_conversation`，用户说再见、退下、没事了这类结束语时调用，
  助手说完告别语就直接回休眠，不再多花一轮请求。

`search_history` 是长期记忆的入口：更早的对话不在模型上下文里，
它必须调这个工具去数据库里翻，拿到的是原始记录而不是凭印象编的答案。

如果用的接口不支持 `tools` 参数，助手会自动去掉工具参数重试一次，
之后按普通对话跑，不会因此罢工。

## 树莓派工具

跑在树莓派上时自动启用（`config.py` 里的 `PI_TOOLS` 设成 `None` 就是自动判断，
也可以在别的机器上设 `True` 强开）：

| 工具 | 干什么 |
|---|---|
| `system_status` | 温度、CPU 负载和频率、内存、磁盘、运行时长、供电与降频记录 |
| `network_status` | 网卡和 IP、默认网关、无线 SSID 与信号、能不能连外网 |
| `gpio_list` | 列出已登记管脚和当前电位 |
| `gpio_read` | 读某个管脚的电位 |
| `gpio_write` | 把某个管脚拉高或拉低 |

信息来自 `/proc`、`/sys` 和 `vcgencmd`，不依赖 `psutil`。
`vcgencmd get_throttled` 的位会翻译成人话，比如"当前供电不足""曾经降过频"。

### GPIO 的安全边界

模型**只能操作 `config.py` 里登记过的管脚**，而且要用名字：

```python
GPIO_PINS = {
    "风扇": 18,
    "补光灯": 17,
}
```

没登记的名字或管脚号一律拒绝，也不提供改管脚模式、不提供任意管脚号写入。
语音指令直接驱动物理设备，让模型随便点一根引脚是危险的，所以这里留了一道闸。
默认是空的，不配置就什么都控制不了。

后端按 `gpiozero`、`RPi.GPIO` 的顺序找，设备对象按管脚缓存，
不会每次调用都重新初始化把已有状态冲掉。两个都没装时工具会明确说明，
不会静默失败。

树莓派上装依赖：

```bash
sudo apt install -y python3-gpiozero
```

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
