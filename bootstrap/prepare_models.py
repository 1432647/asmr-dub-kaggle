"""Resolve every model the pipeline needs: mounted first, downloaded second.

`/kaggle/working` is capped at 20GB and the full model set is ~19GB, so
attaching the weights as a read-only Kaggle Dataset is not an optimisation --
it is what makes the pipeline fit at all.

Read-only mounts create one wrinkle: IndexTTS insists on reading its auxiliary
models from `{model_dir}/hf_cache/`, and we cannot write into `/kaggle/input`.
The fix is a *composed* model directory in scratch made of symlinks to whatever
is mounted, with only the genuinely-missing pieces downloaded. Symlinks work on
Kaggle's overlayfs and cost no space.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asmrdub import pins

BOLD = "\033[1m"
RESET = "\033[0m"


def say(message: str) -> None:
    print(f"{BOLD}[models]{RESET} {message}", flush=True)


# --------------------------------------------------------------------------
# Mount scanning
# --------------------------------------------------------------------------


def _walk_dirs(root: str, max_depth: int = 6):
    """Directories under ``root``, depth-limited.

    Kaggle mounts can be deep and wide; an unbounded walk over several
    multi-gigabyte datasets wastes real time on stat calls.
    """
    if not os.path.isdir(root):
        return
    root_depth = root.rstrip(os.sep).count(os.sep)
    for current, subdirs, _ in os.walk(root):
        if current.count(os.sep) - root_depth >= max_depth:
            subdirs[:] = []
        yield current


def find_indextts_dir(input_root: str) -> str | None:
    """A mounted directory holding the IndexTTS-2.5 checkpoints.

    Identified by the four large files that cannot come from anywhere else.
    """
    markers = ("gpt.pth", "s2mel.pth", "codec.pth", "config.yaml")
    for directory in _walk_dirs(input_root):
        if all(os.path.isfile(os.path.join(directory, name)) for name in markers):
            return directory
    return None


def find_whisper_dir(input_root: str) -> str | None:
    """A CTranslate2 whisper model directory (model.bin + config.json)."""
    for directory in _walk_dirs(input_root):
        if os.path.isfile(os.path.join(directory, "model.bin")) and os.path.isfile(
            os.path.join(directory, "config.json")
        ):
            return directory
    return None


def find_file(input_root: str, filename: str) -> str | None:
    """A mounted file by exact name.

    Only use this for distinctive names. `config.json` lives in every model
    directory ever published, so matching it by basename alone would happily
    link whisper's config as BigVGAN's -- see `find_file_in`.
    """
    for directory in _walk_dirs(input_root):
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def find_file_in(input_root: str, parent: str, filename: str) -> str | None:
    """A mounted file whose immediate parent directory is named `parent`.

    Used for the ambiguous names: `bigvgan/config.json` must come from a
    directory actually called `bigvgan`.
    """
    for directory in _walk_dirs(input_root):
        if os.path.basename(directory.rstrip(os.sep)).lower() != parent.lower():
            continue
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def find_relative(input_root: str, relative: str) -> str | None:
    """Resolve a `hf_cache`-relative path against the mounts.

    `a/b.json` requires a parent directory named `a`; a bare `b.bin` is matched
    by name anywhere. This is the only lookup `compose_aux` should use.
    """
    parts = relative.replace("\\", "/").split("/")
    if len(parts) == 1:
        return find_file(input_root, parts[0])
    return find_file_in(input_root, parts[-2], parts[-1])


def find_gguf(input_root: str, hint: str = "gemma") -> str | None:
    """A mounted GGUF, preferring names containing ``hint`` then the largest."""
    matches = []
    for directory in _walk_dirs(input_root):
        try:
            entries = os.listdir(directory)
        except OSError:
            continue
        for name in entries:
            if name.lower().endswith(".gguf"):
                path = os.path.join(directory, name)
                try:
                    matches.append((os.path.getsize(path), path))
                except OSError:
                    continue
    if not matches:
        return None
    preferred = [m for m in matches if hint.lower() in os.path.basename(m[1]).lower()]
    pool = preferred or matches
    return max(pool)[1]


# --------------------------------------------------------------------------
# Linking / downloading primitives
# --------------------------------------------------------------------------


def link(source: str, target: str) -> None:
    """Symlink ``source`` at ``target``, replacing anything already there.

    Falls back to copying when the filesystem refuses symlinks -- correctness
    first, disk second.
    """
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    if os.path.islink(target) or os.path.isfile(target):
        os.unlink(target)
    elif os.path.isdir(target):
        shutil.rmtree(target)
    try:
        os.symlink(source, target)
    except OSError:
        if os.path.isdir(source):
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def _hf_download(repo: str, dest: str, allow=None, ignore=None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo,
        local_dir=dest,
        allow_patterns=list(allow) if allow else None,
        ignore_patterns=list(ignore) if ignore else None,
        max_workers=4,
    )


def _hf_file(repo: str, filename: str, dest_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir)


def download_url(url: str, target: str, expected_size: int | None = None) -> str:
    """Download with a size check, resuming nothing but never trusting a stub.

    A truncated 335MB checkpoint fails much later inside torch.load with an
    unhelpful error, so verify here where the fix is obvious.
    """
    if os.path.isfile(target) and (
        expected_size is None or os.path.getsize(target) == expected_size
    ):
        return target
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    say(f"downloading {os.path.basename(target)} ...")
    temp = target + ".part"
    with urllib.request.urlopen(url, timeout=120) as response, open(temp, "wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)
    if expected_size is not None and os.path.getsize(temp) != expected_size:
        actual = os.path.getsize(temp)
        os.unlink(temp)
        raise RuntimeError(
            f"{os.path.basename(target)} is {actual} bytes, expected {expected_size}"
        )
    os.replace(temp, target)
    return target


# --------------------------------------------------------------------------
# IndexTTS
# --------------------------------------------------------------------------


def compose_indextts(model_root: str, input_root: str) -> str:
    """Build a writable IndexTTS checkpoints dir from mounts + downloads."""
    dest = os.path.join(model_root, "indextts")
    os.makedirs(dest, exist_ok=True)

    mounted = find_indextts_dir(input_root)
    if mounted:
        say(f"IndexTTS weights mounted at {mounted}")
        for name in os.listdir(mounted):
            if name == "hf_cache":
                continue
            link(os.path.join(mounted, name), os.path.join(dest, name))
    else:
        missing = [
            name
            for name in pins.INDEXTTS_REQUIRED_FILES
            if not os.path.exists(os.path.join(dest, name))
        ]
        if missing:
            say(f"downloading {pins.INDEXTTS_HF_REPO} (~4.3GB, missing {len(missing)} files)")
            _hf_download(
                pins.INDEXTTS_HF_REPO, dest, ignore=pins.INDEXTTS_SKIP_PATTERNS
            )
        else:
            say("IndexTTS weights already present in scratch")

    absent = [
        name
        for name in pins.INDEXTTS_REQUIRED_FILES
        if not os.path.exists(os.path.join(dest, name))
    ]
    if absent:
        raise RuntimeError(f"IndexTTS is missing required files: {', '.join(absent)}")

    compose_aux(dest, input_root, mounted)
    return dest


def compose_aux(model_dir: str, input_root: str, mounted_indextts: str | None) -> None:
    """Populate {model_dir}/hf_cache with the auxiliary models.

    Paths and filenames here are dictated by index-tts's own
    `utils/model_download.ensure_models_available`; deviating means IndexTTS
    silently re-downloads at inference time into a relative ./checkpoints path.
    """
    cache = os.path.join(model_dir, "hf_cache")
    os.makedirs(cache, exist_ok=True)

    mounted_cache = (
        os.path.join(mounted_indextts, "hf_cache") if mounted_indextts else None
    )

    # w2v-bert-2.0: transformers only reads config + preprocessor + safetensors.
    w2v_dest = os.path.join(cache, "w2v-bert-2.0")
    if not os.path.isfile(os.path.join(w2v_dest, "model.safetensors")):
        source = None
        if mounted_cache and os.path.isdir(os.path.join(mounted_cache, "w2v-bert-2.0")):
            source = os.path.join(mounted_cache, "w2v-bert-2.0")
        else:
            # Match on the directory name, not on `model.safetensors` -- that
            # filename also belongs to the MaskGCT codec and to the Qwen
            # emotion model, either of which would load as garbage here.
            for directory in _walk_dirs(input_root):
                base = os.path.basename(directory.rstrip(os.sep)).lower()
                if "w2v-bert" in base and os.path.isfile(
                    os.path.join(directory, "model.safetensors")
                ):
                    source = directory
                    break
        if source:
            say(f"w2v-bert-2.0 mounted at {source}")
            link(source, w2v_dest)
        else:
            say("downloading facebook/w2v-bert-2.0 (~2.3GB)")
            _hf_download("facebook/w2v-bert-2.0", w2v_dest, allow=pins.W2V_BERT_ALLOW)

    simple_targets = (
        ("campplus_cn_common.bin", "funasr/campplus", "campplus_cn_common.bin"),
        (
            "semantic_codec_model.safetensors",
            "amphion/MaskGCT",
            "semantic_codec/model.safetensors",
        ),
        ("bigvgan/config.json", "nvidia/bigvgan_v2_22khz_80band_256x", "config.json"),
        (
            "bigvgan/bigvgan_generator.pt",
            "nvidia/bigvgan_v2_22khz_80band_256x",
            "bigvgan_generator.pt",
        ),
    )
    for relative, repo, remote in simple_targets:
        target = os.path.join(cache, relative)
        if os.path.isfile(target):
            continue
        if mounted_cache:
            candidate = os.path.join(mounted_cache, relative)
            if os.path.isfile(candidate):
                link(candidate, target)
                continue
        # Path-aware: `bigvgan/config.json` must come from a `bigvgan/` dir.
        found = find_relative(input_root, relative)
        if found:
            say(f"{relative} mounted at {found}")
            link(found, target)
            continue
        say(f"downloading {repo}/{remote}")
        staged = _hf_file(repo, remote, os.path.join(cache, "_staging"))
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        shutil.move(staged, target)

    staging = os.path.join(cache, "_staging")
    if os.path.isdir(staging):
        shutil.rmtree(staging, ignore_errors=True)


# --------------------------------------------------------------------------
# Other models
# --------------------------------------------------------------------------


def resolve_whisper(model_root: str, input_root: str) -> str:
    mounted = find_whisper_dir(input_root)
    if mounted:
        say(f"faster-whisper mounted at {mounted}")
        return mounted
    dest = os.path.join(model_root, "faster-whisper-large-v3")
    if os.path.isfile(os.path.join(dest, "model.bin")):
        say("faster-whisper already present in scratch")
        return dest
    say(f"downloading {pins.FASTER_WHISPER_REPO} (~3.1GB)")
    _hf_download(pins.FASTER_WHISPER_REPO, dest)
    return dest


def resolve_hdemucs(model_root: str, input_root: str) -> str:
    name = os.path.basename(pins.HDEMUCS_URL)
    mounted = find_file(input_root, name)
    if mounted and os.path.getsize(mounted) == pins.HDEMUCS_SIZE:
        say(f"HDemucs mounted at {mounted}")
        return mounted
    return download_url(
        pins.HDEMUCS_URL, os.path.join(model_root, name), pins.HDEMUCS_SIZE
    )


def resolve_llm_gguf(input_root: str) -> str | None:
    """A mounted GGUF for the translator, if the user attached one."""
    found = find_gguf(input_root, hint="gemma")
    if found:
        say(f"LLM GGUF mounted at {found} ({os.path.getsize(found) / 1e9:.1f}GB)")
    return found


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def prepare_all(model_root: str, input_root: str) -> dict:
    os.makedirs(model_root, exist_ok=True)
    resolved = {
        "indextts": compose_indextts(model_root, input_root),
        "whisper": resolve_whisper(model_root, input_root),
        "hdemucs": resolve_hdemucs(model_root, input_root),
        "llm_gguf": resolve_llm_gguf(input_root),
    }
    say("all models resolved:")
    for key, value in resolved.items():
        say(f"  {key}: {value}")
    return resolved


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--out-json")
    args = parser.parse_args()
    result = prepare_all(args.model_root, args.input_root)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
