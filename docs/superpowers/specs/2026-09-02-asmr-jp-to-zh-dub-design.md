# ASMR 日→中 配音流水线（Kaggle 双 T4）设计文档

**日期**: 2026-09-02
**目标**: 上传一个十几分钟的多角色日语 ASMR 音频，在网页上一路点到底，得到保留原声优音色与环境音氛围的中文配音，全程有日志和进度可看。

---

## 1. 为什么改造 VideoLingo 而不是自建

VideoLingo（Apache-2.0，18.3k star）已经解决了配音流水线里最脏的三件事：

1. **词级时间戳 → 语义断句**：whisper 词级时间戳 + spaCy 形态学切分 + LLM 语义切分，产出"一句话一行"。
2. **时长受限翻译**：译文预估朗读时长超过原时间轴时，自动让 LLM 缩写；仍超时则变速；仍超时则合并相邻块统一变速。这套时间轴数学（`_8_2_dub_chunks` + `_10_gen_audio.merge_chunks`）自己写要踩很多坑。
3. **可插拔 TTS 后端**：`tts_main` 已有 `custom_tts` 钩子。

我们只替换它的"耳朵"（ASR）、"嘴"（TTS）、和"混音台"，其余原样复用。

上游钉版：
- VideoLingo `814f84eeb98db987510e3558feeb595de2ac328a`
- index-tts `ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c`

钉 commit 是硬要求：上游一改，补丁全废。

## 2. Kaggle 上"跑整个项目"的形态

Kaggle notebook 是一台带 root 的 Linux VM，`git clone` + `pip install` + 后台进程都合法（VideoLingo 官方 Colab notebook 第一格就是 `!git clone`）。所谓"单文件脚本"只是**引导器**：

```
Kaggle cell (粘贴 kaggle_bootstrap.py)
  └─ git clone <overlay 仓库>          # 我们的补丁与新代码
      └─ bootstrap/bootstrap_main.py
          ├─ git clone VideoLingo @钉死commit
          ├─ git clone index-tts   @钉死commit
          ├─ apply_overlay：拷 overlay/ + 打补丁 + 写 config.yaml
          ├─ prepare_models：先探测已挂载 Dataset，缺的才下载
          ├─ prepare_env：建两个 uv venv
          └─ run_all：起 P3 → P2 → P1 → cloudflared → keepalive
```

前置条件：notebook 设置里 **Internet = On**（需手机验证过的账号）+ 加速器 **GPU T4 ×2**。

overlay 仓库也可以打包成 Kaggle Dataset 挂载（`--source dataset`），无需 GitHub。

## 3. 三进程隔离

依赖钉子直接冲突，混装必炸：VideoLingo 要 `librosa==0.11.0` / `opencv==4.11` / `numpy>=2.0.2`；IndexTTS 要 `librosa==0.10.2.post1` / `opencv==4.9.0.80` / `transformers==4.52.1` / `numpy==2.2.6`。因此：

| 进程 | 环境 | 端口 | GPU | 职责 |
|---|---|---|---|---|
| **P1** app | `venv-app`（纯 CPU） | 8501 | — | Streamlit UI、编排、断句、翻译调度、时间轴、混音 |
| **P2** worker | `venv-gpu`（index-tts 的 uv venv） | 127.0.0.1:7861 | cuda:0 | `/asr` faster-whisper、`/separate` HDEMUCS、`/tts` IndexTTS-2.5 |
| **P3** llm | ollama 自带运行时 | 127.0.0.1:11434 | cuda:1 | 翻译 LLM，OpenAI 兼容 `/v1` |

P1 完全不 import torch，所有 GPU 工作走 HTTP 交给 P2。P2 内部单锁串行，模型懒加载。
两个 venv 都用 `uv venv` 独立创建，不污染 Kaggle 基础环境（避免 pip 升级 numpy 把预装包搞坏）。

对外只暴露 8501（cloudflared quick tunnel），带随机密码门。7861/11434 只监听回环。

## 4. 必须偏离上游的地方

### 4.1 换掉 whisperX → faster-whisper

whisperX 硬钉 `torch~=2.8.0` + `pyannote.audio>=4` + `torchcodec`，是 Kaggle 装包最大的爆炸源。
faster-whisper 走 CTranslate2，**依赖里完全没有 torch**（`ctranslate2`/`onnxruntime`/`av`/`tokenizers`），
且 `word_timestamps=True` 产出的结构与 VideoLingo `process_transcription` 期望的一致。

适配层把 faster-whisper 的 `(segments, info)` 转成
`{"segments":[{"start","end","text","words":[{"word","start","end"}]}], "language": info.language}`。

代价：丢掉 whisperX 的 wav2vec2 强制对齐（音素级精修）。对 ASMR 影响有限，因为后续时间轴本身有 tolerance/变速兜底。
**日语耳语/耗音的识别质量是整条链最弱的一环**，换后端救不了——靠第 4.5 节的人工复核环节兜。

模型：`Systran/faster-whisper-large-v3`，`compute_type="float16"`（T4 = sm_75 原生支持 fp16；
注意 `torch.cuda.is_bf16_supported()` 在 torch 2.8 上对 T4 返回 True 是**模拟**支持，不能据此选 bf16）。

### 4.2 原声单独跑全采样率人声分离

上游 `_2_asr` 先把输入压成 **16k 单声道 32kbps mp3**，再喂 Demucs。拿这个结果当克隆参考和环境音垫底是自残——
ASMR 常是双耳（binaural）录音，压成单声道等于把整个卖点删掉。

改为独立一路：原始文件 → torchaudio `HDEMUCS_HIGH_MUSDB_PLUS`（44.1kHz 立体声）→
`vocal_hifi.wav` / `background_hifi.wav`。克隆参考和最终混音都取自这一路；
16k 单声道那一路只留给 ASR。

用 torchaudio 内置 HDEMUCS 而不是 `demucs` pip 包：省一个依赖，权重是 `download.pytorch.org` 上的
335MB `.pt`，且不引入 `demucs` 对 torchaudio 版本的钉子。
15 分钟 44.1k 立体声 = 4000 万采样点，必须分块 → **重叠相加 + 线性交叉淡化**（torchaudio 官方配方），
每块 10s、重叠 10%，并按整轨 mean/std 归一化。

上游 `demucs_vl.demucs_audio()` 被 overlay 替换为调用 P2 `/separate` 的同名函数，
既满足 `_9_refer_audio` 的调用约定，又让 P1 彻底不需要 torch。

### 4.3 分角色：不做说话人聚类

`_9_refer_audio` 按时间轴把人声轨切成 `refers/{n}.wav`，每句中文用**原片那一句**的声音克隆——
谁说的就跟谁的音色，天然多角色。整个 diarization 模块不需要。

新增问题：短句（< 3s）克隆会飘。解法是**参考片段扩窗**：
以该句时间轴为种子，向静默间隔较小的一侧扩展（跨过间隔 ≤ 1.2s 的相邻句，视为同说话人续说），
直到 ≥ 3s 或无法再扩，上限 10s（IndexTTS 内部也会把 prompt 截到 15s）。
纯函数 `expand_window`，可单测。

### 4.4 自写最终混音

上游 `_11_merge_audio` 输出 **16k 单声道 64kbps mp3**，`_12_dub_to_vid` 对纯音频输入直接跳过背景混合——
两条都不能用。替换为 44.1kHz 立体声混音：

1. 画布 = 44.1k 立体声，长度取 `max(背景音长度, 最后一句结束 + 尾巴)`
2. 每段 TTS（22.05k 单声道，经 ffmpeg 变速后仍是 22.05k）→ ffmpeg 重采样到 44.1k
3. **摆位跟随原声**：量原始人声该段的 L/R 能量比得到 pan ∈ [-1,1]，用等功率（cos/sin）声像律把单声道配音摆到同一位置。这是 ASMR 的命门——原声在右耳说话，中文配音也必须在右耳。
4. 叠加 `background_hifi.wav`（已去人声的环境音）原样保留
5. 峰值超过 1.0 时整体线性缩放（不做压缩器，避免改变 ASMR 的动态）
6. 输出 44.1k 16-bit 立体声 WAV + 192kbps MP3 + `dub.srt`

### 4.5 UI 增两个面板

- **日志尾巴**：上游日志只进 notebook stdout，网页上看不到。用它现成的 `@st.fragment(run_every=...)` 轮询日志文件尾部。
- **TTS 前可编辑复核表**：`st.data_editor` 展示 `序号/起止/时长/日文/中文/预估时长`，改完保存回 xlsx 再点合成。默认开启（可关）。

配套改动：上游 `_8_2_dub_chunks` 会把 `tts_tasks.text` 与 `output/trans.srt` 的行做**精确串接匹配**，
匹配不上就 `raise ValueError("Matching failed")`。人工改了译文必然匹配失败。
替换为 1:1 映射（`lines=[text]`, `src_lines=[origin]`）——该匹配逻辑本来只为把一行配音拆成多行显示字幕，
对配音无意义，去掉同时消灭一个脆弱失败模式。

### 4.6 不加载 QwenEmotion（情感描述文本控制）

每句的参考音频就是原声那一句。IndexTTS-2.5 在未给 `emo_vector` 时会把音色参考**同时**当情感参考
（`infer_generator` 里 `emo_audio_prompt = spk_audio_prompt`），原声优的呼吸、颤音、气声直接迁移。
QwenEmotion 是"从中文文本猜情感向量"，严格更差；且一旦给了 `emo_vector` 就会把
`emo_audio_prompt` 置 None，**关掉**参考音频的情感通道。
故 `use_qwen_emo=False`，并跳过 `qwen0.6bemo4-merge/` 的 1.2GB 权重下载。腾出的 GPU1 给翻译 LLM。

## 5. 翻译 LLM：Gemma 4 12B 无审查

`gemma4` 家族五个尺寸：E2B(2.3B eff) / E4B(4.5B eff) / 12B unified(11.95B dense) / 26B-A4B(MoE) / 31B dense。

选 **12B unified**：
- 26B-A4B 的 Q4_K_M 是 16.8GB，单张 T4（15GB）装不下；用两张就跟 TTS 抢卡
- 31B dense 更不可能
- E4B 只有 4.5B effective，日→中文学性翻译质量不够
- 12B dense Q4_K_M = 7.4GB，单卡舒适，256K 上下文

无审查版本：`zaakirio/gemma-4-12b-it-uncensored-GGUF:Q4_K_M`（Heretic 去审查，README 明确 12B unified，
27.8k 下载）。ASMR 题材必须去审查，否则翻译中途拒答会让 `ask_gpt` 的 5 次重试全废。
回退：ollama 官方 `gemma4:12b`（有审查，仅保证能跑）。

运行时用 **ollama**（`ollama-linux-amd64.tar.zst`，1.42GB，自带 CUDA 运行时）：
- 单个 tarball，零 pip 依赖冲突（llama.cpp 官方 release **没有** linux CUDA 预编译产物，
  只有 CPU/vulkan/sycl/rocm；从源码编译要 20-30 分钟）
- 开箱 OpenAI 兼容 `/v1`，VideoLingo 的 `ask_gpt` 只认这个
- `CUDA_VISIBLE_DEVICES=1` 锁 GPU1，`OLLAMA_MODELS` 指到 scratch 分区避开 20GB 配额

VideoLingo 配置改动（省时间）：
- `reflect_translate: false` — 上游默认 true 会跑"直译 + 反思意译"两遍 LLM，本地 12B 上翻倍到 40+ 分钟
- `max_workers: 2` + `OLLAMA_NUM_PARALLEL=2` — 7.4GB 权重后还剩 ~7GB 放 2 路 KV cache
- `api.key: "ollama"`（非空即可）、`api.base_url: "http://127.0.0.1:11434/v1"`（含 `v1`，`ask_gpt` 不会再拼）

## 6. 权重与磁盘

`/kaggle/working` 硬上限 20GB。总权重约 19GB，**必须**走 Dataset 只读挂载（`/kaggle/input` 不占配额）。
`prepare_models` 先递归探测 `/kaggle/input`，缺什么才下载，下载落到 scratch。

| 用途 | 来源 | 大小 |
|---|---|---|
| TTS 主权重 | `IndexTeam/IndexTTS-2.5`（allow-list 排除 `qwen0.6bemo4-merge/`） | 4.28 GB |
| TTS 辅助 | `facebook/w2v-bert-2.0`(仅 model.safetensors+config) / `amphion/MaskGCT` semantic_codec / `funasr/campplus` / `nvidia/bigvgan_v2_22khz_80band_256x` | 2.98 GB |
| ASR | `Systran/faster-whisper-large-v3` | 3.09 GB |
| 人声分离 | `hdemucs_high_trained.pt` | 0.33 GB |
| 翻译 | gemma-4-12b Q4_K_M GGUF | 7.38 GB |
| LLM 运行时 | ollama tarball（解压后） | ~1.4 GB |

IndexTTS 的辅助权重路径是**硬约定**：`infer_v2_5.py` 在 import 时执行
`os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'`（相对路径！），且从
`{model_dir}/hf_cache/{w2v-bert-2.0, campplus_cn_common.bin, bigvgan/, semantic_codec_model.safetensors}`
读取。因此 worker **必须 chdir 到 index-tts 仓库根**，并把 `checkpoints` 软链到解析出的模型目录。

音频输入也走 Dataset 挂载（十几分钟的文件经 cloudflared 快隧道上传容易断）；
浏览器 `st.file_uploader` 作为备用。

## 7. 时间预算

| 阶段 | 时长 |
|---|---|
| 环境准备（首次，含 torch cu128 下载） | 8-14 min |
| 权重（已挂 Dataset / 需下载） | 0.5 min / 15-25 min |
| ASR（15 min 音频，large-v3 fp16） | 3-6 min |
| HDEMUCS 44.1k 分离 | 2-4 min |
| 断句 + 翻译（本地 12B，单遍，2 并发） | 15-25 min |
| TTS（~250 句） | 20-45 min |
| 混音 | 1-2 min |
| **合计** | **约 1.2-2 h**，单会话 9-12h 额度充裕 |

## 8. 安全

- cloudflared quick tunnel 的 URL **公开可访问**，随机 16 字符密码门是唯一防线；密码在 notebook 输出里打印
- 7861/11434 仅绑 127.0.0.1，不出隧道
- 无鉴权 GPU 服务只在容器内可达

## 9. 文件结构

```
asmr-dub-kaggle/
  kaggle_bootstrap.py              # 唯一需要粘贴进 Kaggle 的文件
  asmrdub/                         # 纯逻辑，无 torch / 无 VideoLingo 依赖，可本地单测
    pins.py            上游 commit、模型仓库、端口、路径
    chunking.py        重叠相加分块规划
    refer_window.py    参考片段扩窗
    mixdown.py         声像估计 / 等功率摆位 / 混音规划
    asr_format.py      faster-whisper → VideoLingo segments
    srt_time.py        时间字符串 ↔ 秒
    review_table.py    复核表 ↔ tts_tasks.xlsx 往返
    vl_config.py       VideoLingo config.yaml 覆盖生成
    patches.py         对上游的补丁定义
  overlay/                         # 拷进 VideoLingo 树的适配层
    core/asr_backend/faster_whisper_local.py
    core/asr_backend/demucs_vl.py          （替换：转发到 worker /separate）
    core/asr_backend/asr_client.py
    core/tts_backend/custom_tts.py         （替换：IndexTTS 客户端）
    core/_2_asr_asmr.py
    core/_8_2_dub_chunks_asmr.py
    core/_9_refer_audio_hifi.py
    core/_11_merge_audio_asmr.py
    asmr_ui.py                             （Streamlit 主界面）
  worker/server.py                 # P2
  bootstrap/{bootstrap_main,prepare_env,prepare_models,apply_overlay}.py
  runtime/{ollama_svc,tunnel,run_all}.py
  tests/                           # pytest，纯 CPU，本地可跑
```

## 10. 已知风险

| 风险 | 缓解 |
|---|---|
| 日语耳语 ASR 错得离谱 | 人工复核表（默认开启） |
| ollama 可能不认第三方 Gemma4Unified GGUF | 自动回退官方 `gemma4:12b` 并在 UI 报警 |
| `uv sync` 拉 torch cu128 约 2.5GB，网络波动 | 重试 3 次；失败给出明确报错而不是静默降级 |
| 短句克隆音色飘 | 扩窗；仍不足则回退到全片最长的那句作参考 |
| Kaggle 会话被回收 | 中间产物全在 `/kaggle/working/VideoLingo/output`，`@check_file_exists` 让每阶段可续跑 |
