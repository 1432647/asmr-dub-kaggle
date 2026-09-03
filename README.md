# ASMR 日→中 配音（Kaggle 双 T4）

上传一段日语 ASMR 音频，在网页上点几下，得到**保留原声优音色和环境音氛围**的中文配音。
全程有日志和进度可看，识别结果可人工复核后再合成。

从 [VideoLingo](https://github.com/Huanshere/VideoLingo) 抽出音频翻译配音链路做特化改造：
TTS 换成 [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)（零样本音色克隆），
翻译用本地无审查 Gemma 4 12B，人声分离改成全采样率立体声。

---

## 快速开始

1. 新建 Kaggle Notebook，**Settings → Accelerator = GPU T4 x2**、**Internet = On**
   （Internet 需要手机验证过的账号）
2. 左侧 **Data → Add Input**，挂载：
   - 你要转换的音频所在的 Dataset（十几分钟的文件走浏览器上传容易断）
   - 权重 Dataset（可选但强烈建议，见下方「权重」）
3. 新建一个 cell，粘贴 [`kaggle_bootstrap.py`](kaggle_bootstrap.py) 全部内容，运行
4. 等横幅打印出网址和密码，打开网址

```
====================================================================
网页地址: https://xxxx-yyyy-zzzz.trycloudflare.com
访问密码: Kj3mP9xQr2vB
====================================================================
```

> ⚠️ 这个地址是**公开可访问**的，密码是唯一防线。用完请停止会话。

## 网页上的四步

| 步骤 | 做什么 | 耗时 |
|---|---|---|
| ① 选择音频 | 从挂载的 Dataset 下拉选，或直接上传 | — |
| ② 识别与翻译 | 语音识别 → 人声分离 → 断句 → 翻译 → 时间轴对齐 | 20-35 分钟 |
| ③ 人工复核 | 可编辑表格：日文识别结果 / 中文译文，改完再合成 | 看你 |
| ④ 合成配音 | 逐句克隆 → 混音 → 下载 | 20-45 分钟 |

每一步的中间产物都在 `output/`，崩了可以从同一步续跑。会话被回收后重新运行引导 cell 即可。

## 产出

```
output/
  dub_44k_stereo.wav    最终成品：44.1kHz 16-bit 立体声
  dub.mp3               同上，192kbps
  dub.srt               中文字幕（配音后的实际时间轴）
  mix_report.json       混音报告：峰值、增益、声像统计
  src.srt / trans.srt   日文原文 / 中文译文字幕
  audio/
    vocal_hifi.wav      分离出的人声（全采样率立体声）
    background_hifi.wav 分离出的环境音
    refers/             每句的克隆参考片段 + index.json（含声像位置）
    tts_tasks.xlsx      配音任务表（复核表就是它）
```

## 三个关键设计

**多角色不需要说话人分离。** 每句中文都从**原片那一句**的声音克隆——谁说的就跟谁的音色。
短句（<3s）参考不够会向邻近句扩窗，但只跨过 ≤1.2s 的静默：更长的间隔通常意味着换人了，
拼进来只会把两个人的音色混成一个参考。

**声像跟随原声。** ASMR 通常是双耳录音，人声在左右耳之间移动本身就是内容。
流水线量出原声每句在立体声场里的位置（能量比 → 等功率声像律），把中文配音摆到同一位置。
原声在右耳说话，中文配音也在右耳。

**不用「情感描述文本控制」。** 每句的参考音频就是原声那一句，IndexTTS 在没给 `emo_vector` 时
会把音色参考同时当情感参考，原声优的呼吸、颤音、气声直接迁移。
从中文文本猜情感严格更差，而且给了 `emo_vector` 反而会**关掉**参考音频的情感通道。

## 权重（约 19GB）

`/kaggle/working` 硬上限 20GB，所以**必须**把权重挂成只读 Dataset（`/kaggle/input` 不占配额）。
`prepare_models` 会先递归探测 `/kaggle/input`，缺什么才下载。

| 用途 | 来源 | 大小 |
|---|---|---|
| TTS | `IndexTeam/IndexTTS-2.5`（跳过 `qwen0.6bemo4-merge/`） | 4.3 GB |
| TTS 辅助 | w2v-bert-2.0 / MaskGCT semantic codec / CAMPPlus / BigVGAN | 3.0 GB |
| 语音识别 | `Systran/faster-whisper-large-v3` | 3.1 GB |
| 人声分离 | torchaudio `hdemucs_high_trained.pt` | 0.33 GB |
| 翻译 | `zaakirio/gemma-4-12b-it-uncensored-GGUF:Q4_K_M` | 7.4 GB |

首次全下载约 15-25 分钟；权重已挂载则跳过。

Gemma 4 家族里选 **12B unified** 的理由：26B-A4B 的 Q4_K_M 是 16.8GB，单张 T4（15GB）装不下，
用两张就得跟 TTS 抢卡；E4B 只有 4.5B effective，翻译质量不够；12B dense 刚好，256K 上下文。
用去审查版本是因为 ASMR 题材一旦被拒答，`ask_gpt` 重试 5 次后整个翻译阶段就崩了。

## 架构

三个进程，因为依赖钉子直接冲突（VideoLingo 要 librosa 0.11 / opencv 4.11，
IndexTTS 要 librosa 0.10.2 / opencv 4.9 / transformers 4.52.1，混装必炸）：

```
              cloudflared quick tunnel
                        │
                   :8501 (仅此端口对外)
┌───────────────────────┴────────────────────────┐
│ P1  Streamlit UI + 编排        venv-app / CPU  │
│     断句 · 翻译调度 · 时间轴 · 混音（不 import torch）│
└──────┬──────────────────────────┬──────────────┘
       │ HTTP 127.0.0.1:7861      │ OpenAI /v1 127.0.0.1:11434
┌──────┴───────────────┐   ┌──────┴──────────────────┐
│ P2  GPU worker       │   │ P3  ollama              │
│     cuda:0           │   │     cuda:1              │
│  /asr  faster-whisper│   │  Gemma 4 12B 无审查     │
│  /separate  HDemucs  │   │                         │
│  /tts  IndexTTS-2.5  │   │                         │
│  单锁串行 + 懒加载    │   │                         │
└──────────────────────┘   └─────────────────────────┘
```

P2/P3 只监听回环，**没有任何鉴权**，绝不能暴露到隧道。

## 目录结构

```
kaggle_bootstrap.py   唯一需要粘贴进 notebook 的文件
asmrdub/              纯逻辑核心（不 import torch/numpy/streamlit，笔记本上可测）
bootstrap/            装配期：clone 上游、打补丁、解析权重、建 venv
runtime/              运行期：起服务、ollama、cloudflared 隧道
worker/               GPU worker（在 index-tts 的 venv 里跑）
overlay/              覆盖到 VideoLingo 之上的文件
tests/                312 个离线测试
```

> `bootstrap/` 不叫 `setup/`：Kaggle 镜像自带一个已安装的 `setup` 发行版
> （`dist-packages/setup/__init__.py`），同名目录会被它遮蔽。同理每个被 import
> 的目录都有 `__init__.py`——命名空间包永远输给已安装的常规包，`sys.path` 顺序
> 救不了。`tests/test_package_layout.py` 用诱饵包复现了这个失败并守住它。

## 相对上游 VideoLingo 的改动

上游钉死 commit（`814f84ee`），改动方式优先「新增 overlay 文件」而非「改上游」——
新文件不会因上游变动而失效。只有 6 处必须动上游函数内部的补丁，都在
[`asmrdub/patches.py`](asmrdub/patches.py) 里带原因说明，锚点找不到就**报错退出**而不是猜。

| 改动 | 为什么 |
|---|---|
| whisperX → faster-whisper | whisperX 硬钉 torch~=2.8 + pyannote + torchcodec，装包最大爆炸源；faster-whisper 走 CTranslate2 完全不依赖 torch |
| 原声单独跑 44.1kHz 立体声 HDemucs | 上游先把输入压成 16k 单声道 32kbps 再分离，双耳录音直接被毁 |
| 自写混音 | 上游输出 16k 单声道 64kbps，且纯音频输入时直接跳过环境音混合 |
| `_8_2_dub_chunks` 换成 1:1 映射 | 上游把配音文本与 `trans.srt` 做精确串接匹配，匹配不上就 `raise`——人工改译文必然触发 |
| `custom_tts` 收到句号和任务表 | 需要句号才能找到该句的参考音频（否则只能从临时文件名反解，脆） |
| g2p_en 改懒加载 | 构造估算器时会下载 nltk 数据；纯中日流程根本用不到英文音节 |
| TTS 串行 | worker 单锁，并发只增加排队和超时风险 |
| `real_dur` 初始化成 float | 上游写 `= 0` 得到 int64 列，再塞 2.78 进去；pandas 3 直接报错，配音第一句就崩（另把 `pandas>=2.2.3` 收紧成 `<3`） |

## 本地开发

纯逻辑核心（`asmrdub/`）不 import torch / numpy / streamlit / VideoLingo，笔记本上就能跑：

```bash
uv venv .venv --python 3.11
uv pip install --python .venv pytest
.venv/bin/python -m pytest        # 纯逻辑单测
```

集成测试需要一份打好补丁的 VideoLingo，上游合约测试还需要 index-tts：

```bash
# 拉取钉死的上游（两个）
for spec in "Huanshere/VideoLingo 814f84eeb98db987510e3558feeb595de2ac328a /tmp/vl" \
            "index-tts/index-tts ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c /tmp/it"; do
  set -- $spec
  git init -q "$3" && git -C "$3" remote add origin "https://github.com/$1.git" \
    && git -C "$3" fetch --depth 1 origin "$2" \
    && git -C "$3" checkout --force FETCH_HEAD
done

# 打补丁 + 覆盖 overlay
python bootstrap/apply_overlay.py --repo-root /tmp/vl --overlay-root overlay

# 跑全部测试（含集成）
uv pip install --python .venv pandas openpyxl numpy soundfile rich openai \
  json-repair autocorrect-py pydub syllables pypinyin requests edge-tts \
  faster-whisper "spacy>=3.8.7,<3.9" torch torchaudio \
  https://github.com/explosion/spacy-models/releases/download/ja_core_news_md-3.8.0/ja_core_news_md-3.8.0-py3-none-any.whl
ASMRDUB_VL_ROOT=/tmp/vl ASMRDUB_IT_ROOT=/tmp/it .venv/bin/python -m pytest
```

五类测试，全部离线、无 GPU、不联网：

- **纯逻辑单测**：扩窗、声像互逆、重叠相加、SRT 时间、复核表往返、盘位选择
- **包布局回归**：用诱饵包复现 Kaggle 上 `setup` 被遮蔽的失败，守住命名与 `__init__.py`
- **合约测试**（AST 解析上游源码）：我们传的每个关键字参数在钉死的上游里真实存在
- **假服务集成**：假 GPU worker（正弦波代替 TTS）+ 假 OpenAI 端点（按 prompt 判断阶段返回对应 schema），跑的是真的 VideoLingo 阶段代码
- **CPU torch**：用替身分离模型跑真的 `do_separate`，断言直通模型能精确重建输入

`ASMRDUB_VL_ROOT` / `ASMRDUB_IT_ROOT` 未设时对应测试自动跳过，裸克隆下 `pytest` 仍全绿。

## 已知限制

- **日语耳语/气声的识别是整条链最弱的一环。** VAD 阈值已经调低（0.25），
  但 ASR 错了后面全错——这就是第 ③ 步人工复核存在的原因，默认开启。
- cloudflared 快隧道 URL 公开，只有密码保护。
- 单句时长塞不进原时间轴时会变速，上限 1.20（可在侧栏调）；ASMR 加速过头会失真，所以比上游的 1.4 保守。
- 会话额度 9-12 小时，全流程约 1.2-2 小时，够用但别拖。

## 许可

本项目代码 MIT。上游各自的许可各自适用：
VideoLingo Apache-2.0，IndexTTS-2.5 bilibili 模型许可协议，Gemma 4 Apache-2.0（权重按 Gemma 许可）。

克隆声音需要征得被克隆者同意——IndexTTS 不会替你检查这件事。
