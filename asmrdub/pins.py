"""Pinned upstream revisions, model sources, ports and paths.

Everything version-sensitive lives here so a single edit re-pins the whole
pipeline. Upstream commits are pinned deliberately: this project patches
VideoLingo's internals, and an upstream refactor would silently invalidate
those patches.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Upstream repositories (pinned)
# --------------------------------------------------------------------------

VIDEOLINGO_REPO = "https://github.com/Huanshere/VideoLingo.git"
VIDEOLINGO_COMMIT = "814f84eeb98db987510e3558feeb595de2ac328a"

INDEXTTS_REPO = "https://github.com/index-tts/index-tts.git"
INDEXTTS_COMMIT = "ee40fa7d6c6b8a2c7f06105f9f1e65775b74868c"

# --------------------------------------------------------------------------
# Model sources
# --------------------------------------------------------------------------

# IndexTTS-2.5 main weights. qwen0.6bemo4-merge/ is deliberately excluded:
# it is only needed for use_qwen_emo=True (text->emotion guessing), which we
# never enable because every line already has real reference audio.
INDEXTTS_HF_REPO = "IndexTeam/IndexTTS-2.5"
INDEXTTS_SKIP_PATTERNS = ("qwen0.6bemo4-merge/*",)

# Files that must exist for IndexTTS2 v2.5 to construct.
INDEXTTS_REQUIRED_FILES = (
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
)

# Auxiliary models. IndexTTS reads these from {model_dir}/hf_cache/<dest>
# with hardcoded names -- see indextts/utils/model_download.py.
AUX_MODELS = (
    # (kind, hf_repo, remote_path, dest_relative_to_hf_cache)
    ("dir", "facebook/w2v-bert-2.0", None, "w2v-bert-2.0"),
    ("file", "amphion/MaskGCT", "semantic_codec/model.safetensors",
     "semantic_codec_model.safetensors"),
    ("file", "funasr/campplus", "campplus_cn_common.bin", "campplus_cn_common.bin"),
    ("file", "nvidia/bigvgan_v2_22khz_80band_256x", "config.json", "bigvgan/config.json"),
    ("file", "nvidia/bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt",
     "bigvgan/bigvgan_generator.pt"),
)

# Only these files are needed from the w2v-bert-2.0 repo; conformer_shaw.pt
# is a 2.3GB fairseq2 checkpoint transformers never reads.
W2V_BERT_ALLOW = ("config.json", "preprocessor_config.json", "model.safetensors")

# ASR. faster-whisper (CTranslate2) has zero torch dependency, unlike whisperX.
FASTER_WHISPER_REPO = "Systran/faster-whisper-large-v3"

# Source separation: torchaudio's bundled HDemucs. Avoids the `demucs` pip
# package and its torchaudio version pins.
HDEMUCS_URL = "https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt"
HDEMUCS_SIZE = 334697255
HDEMUCS_SR = 44100
HDEMUCS_SOURCES = ("drums", "bass", "other", "vocals")

# Translation LLM. Gemma 4 12B "Unified" (11.95B dense, 256K ctx), Heretic
# de-censored. 12B is the largest family member whose Q4_K_M fits one T4
# alongside nothing else; 26B-A4B's Q4_K_M is 16.8GB.
OLLAMA_MODEL = "hf.co/zaakirio/gemma-4-12b-it-uncensored-GGUF:Q4_K_M"
OLLAMA_MODEL_FALLBACK = "gemma4:12b"
OLLAMA_TARBALL = "https://ollama.com/download/ollama-linux-amd64.tar.zst"
# Verified against a Range request; a truncated download otherwise looks
# "already fetched" on the next run and fails at extraction instead.
OLLAMA_TARBALL_SIZE = 1422262024

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)

# --------------------------------------------------------------------------
# Ports
# --------------------------------------------------------------------------

UI_PORT = 8501       # Streamlit, the only port exposed through the tunnel
WORKER_PORT = 7861   # GPU worker, loopback only
OLLAMA_PORT = 11434  # LLM, loopback only

# --------------------------------------------------------------------------
# GPU assignment
# --------------------------------------------------------------------------

WORKER_GPU = "0"  # ASR + separation + TTS
LLM_GPU = "1"     # ollama

# --------------------------------------------------------------------------
# Audio / pipeline constants
# --------------------------------------------------------------------------

TTS_SR = 22050        # IndexTTS-2.5 output sample rate
MIX_SR = 44100        # final mixdown sample rate
ASR_SR = 16000        # what whisper wants

# Reference-clip window expansion (see asmrdub.refer_window)
REFER_MIN_SEC = 3.0
REFER_MAX_SEC = 10.0
REFER_JOIN_GAP = 1.2

# Overlap-add chunking for HDemucs (see asmrdub.chunking)
SEP_CHUNK_SEC = 10.0
SEP_OVERLAP = 0.1

LANG_JA = "ja"
LANG_ZH = "zh"
INDEXTTS_LANG = "ZH"  # lang code passed to IndexTTS2.infer for the output


def kaggle_dirs() -> dict[str, str]:
    """Resolve Kaggle-ish directories, with local fallbacks for testing.

    Scratch selection measures actual free space rather than trusting a fixed
    path order. On Kaggle `/kaggle/working` is capped at 20GB while `/kaggle/temp`
    sits on the larger container disk, but which paths exist and how much room
    each has varies between images -- and the model set is ~19GB, so guessing
    wrong means running out of disk 20 minutes in.
    """
    on_kaggle = os.path.isdir("/kaggle")
    if on_kaggle:
        working = "/kaggle/working"
        inputs = "/kaggle/input"
        scratch_candidates = ["/kaggle/temp", "/kaggle/tmp", "/tmp", working]
    else:
        base = os.environ.get("ASMRDUB_HOME", os.path.expanduser("~/asmrdub"))
        working = os.path.join(base, "working")
        inputs = os.path.join(base, "input")
        scratch_candidates = [os.path.join(base, "scratch")]

    override = os.environ.get("ASMRDUB_SCRATCH")
    if override:
        scratch_candidates = [override, *scratch_candidates]

    scratch = pick_scratch(scratch_candidates)
    return {
        "working": working,
        "input": inputs,
        "scratch": scratch,
        "on_kaggle": on_kaggle,
    }


def free_gb(path: str) -> float:
    """Free space in GB on the filesystem holding `path`, or -1.0 if unknown."""
    try:
        stat = os.statvfs(path)
    except (OSError, AttributeError):   # AttributeError: Windows
        try:
            import shutil

            return shutil.disk_usage(path).free / 1e9
        except OSError:
            return -1.0
    return (stat.f_bavail * stat.f_frsize) / 1e9


def pick_scratch(candidates, need_gb: float = 30.0) -> str:
    """Pick where to put ~30GB of models and virtualenvs.

    Preference order is meaningful (`/kaggle/temp` before `/tmp`, which may be a
    RAM-backed tmpfs on some images), so this takes the *first* candidate with
    enough room rather than always the largest. If nothing has `need_gb`, it
    falls back to the roomiest so the run gets as far as it can and fails with a
    real ENOSPC rather than a silently bad choice.
    """
    writable = [(candidate, free_gb(candidate))
                for candidate in candidates if _writable(candidate)]
    if not writable:
        return candidates[-1]
    for candidate, free in writable:
        if free >= need_gb:
            return candidate
    return max(writable, key=lambda item: item[1])[0]


def _writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".asmrdub_write_probe")
        with open(probe, "w") as fh:
            fh.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False
