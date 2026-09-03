"""Full setup, driven by the single-file bootstrap cell.

Clones the two pinned upstreams, applies the overlay, resolves the models,
builds both virtualenvs, fetches the standalone binaries, and writes the state
file that `runtime/run_all.py` consumes.

Each step is idempotent, so re-running after a failure resumes rather than
starting over -- which matters when a step is a 4GB download.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from asmrdub import pins  # noqa: E402
from bootstrap import prepare_env, prepare_models  # noqa: E402

BOLD = "\033[1m"
CYAN = "\033[36m"
RESET = "\033[0m"

STATE_FILE = "asmrdub_state.json"


def say(message: str) -> None:
    print(f"{CYAN}{BOLD}[setup]{RESET} {message}", flush=True)


def step(number: int, total: int, title: str) -> None:
    print(f"\n{CYAN}{BOLD}=== [{number}/{total}] {title} ==={RESET}", flush=True)


def clone_pinned(repo: str, commit: str, dest: str) -> str:
    """Fetch exactly one commit.

    `git clone --depth 1` cannot target a commit, and a full clone of
    VideoLingo plus index-tts is hundreds of megabytes of history nobody reads.
    init + fetch --depth 1 <sha> gets one commit's tree.
    """
    if os.path.isdir(os.path.join(dest, ".git")):
        current = subprocess.run(
            ["git", "-C", dest, "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        if current == commit:
            say(f"{os.path.basename(dest)} already at {commit[:8]}")
            return dest
        say(f"{os.path.basename(dest)} at {current[:8]}, re-pinning to {commit[:8]}")
    else:
        os.makedirs(dest, exist_ok=True)
        prepare_env.run(["git", "init", "-q", dest], what="git init")
        prepare_env.run(
            ["git", "-C", dest, "remote", "add", "origin", repo], what="git remote"
        )
    prepare_env.run(
        ["git", "-C", dest, "fetch", "--depth", "1", "origin", commit],
        retries=2, what="git fetch",
    )
    prepare_env.run(
        ["git", "-C", dest, "checkout", "-q", "--force", commit], what="git checkout"
    )
    say(f"{os.path.basename(dest)} pinned at {commit[:8]}")
    return dest


def check_environment() -> None:
    """Fail fast on the two things that make everything else pointless."""
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            "nvidia-smi failed: no GPU. In the notebook sidebar set "
            "Accelerator to 'GPU T4 x2', then restart the session."
        )
    gpus = [line for line in result.stdout.splitlines() if line.startswith("GPU ")]
    say(f"{len(gpus)} GPU(s):")
    for line in gpus:
        say("  " + line)
    if len(gpus) < 2:
        say(
            "WARNING fewer than 2 GPUs. The translator will share a card with "
            "TTS and may run out of memory. 'GPU T4 x2' is the right setting."
        )
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH; it is required for all audio I/O")
    try:
        import urllib.request

        urllib.request.urlopen("https://huggingface.co", timeout=15)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"no internet access ({exc}). In the notebook sidebar turn Internet "
            "On (requires a phone-verified account)."
        ) from exc
    say("environment checks passed")


def free_gb(path: str) -> float:
    """Free space in GB. Shares one implementation with the scratch picker."""
    return max(0.0, pins.free_gb(path))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--password", default="")
    parser.add_argument("--model", default=pins.OLLAMA_MODEL)
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    total = 7
    dirs = pins.kaggle_dirs()
    working, scratch, input_root = dirs["working"], dirs["scratch"], dirs["input"]
    os.makedirs(working, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)

    step(1, total, "环境检查")
    if not args.skip_checks:
        check_environment()
    say(f"working: {working} ({free_gb(working):.0f} GB free)")
    say(f"scratch: {scratch} ({free_gb(scratch):.0f} GB free)")
    say(f"input:   {input_root}")
    if free_gb(scratch) < 25:
        say(
            "WARNING less than 25GB free on scratch. Mount the weights as a "
            "Kaggle Dataset to avoid downloading them into this quota."
        )

    step(2, total, "拉取上游代码（钉死 commit）")
    repo_root = clone_pinned(
        pins.VIDEOLINGO_REPO, pins.VIDEOLINGO_COMMIT,
        os.path.join(working, "VideoLingo"),
    )
    index_tts_root = clone_pinned(
        pins.INDEXTTS_REPO, pins.INDEXTTS_COMMIT,
        os.path.join(working, "index-tts"),
    )

    step(3, total, "打补丁 + 覆盖 overlay + 写配置")
    prepare_env.run(
        [
            sys.executable, os.path.join(HERE, "apply_overlay.py"),
            "--repo-root", repo_root,
            "--overlay-root", os.path.join(ROOT, "overlay"),
            "--ollama-model", args.model,
            "--llm-base-url", f"http://127.0.0.1:{pins.OLLAMA_PORT}/v1",
        ],
        what="apply_overlay",
    )

    step(4, total, "解析模型权重（优先已挂载的 Dataset）")
    models = prepare_models.prepare_all(os.path.join(scratch, "models"), input_root)

    step(5, total, "构建两个 Python 环境")
    uv = prepare_env.ensure_uv()
    app_python = prepare_env.build_app_env(
        uv, os.path.join(scratch, "venv-app"), repo_root
    )
    gpu_python = prepare_env.build_gpu_env(uv, index_tts_root)

    step(6, total, "验证 GPU 环境")
    gpu_info = prepare_env.verify_gpu_env(gpu_python)

    step(7, total, "下载独立二进制（ollama / cloudflared）")
    ollama_binary = prepare_env.install_ollama(scratch)
    cloudflared_binary = prepare_env.install_cloudflared(scratch)

    state = {
        "repo_root": repo_root,
        "index_tts_root": index_tts_root,
        "app_python": app_python,
        "gpu_python": gpu_python,
        "ollama_binary": ollama_binary,
        "cloudflared_binary": cloudflared_binary,
        "ollama_model": args.model,
        "models": models,
        "scratch": scratch,
        "working": working,
        "input_root": input_root,
        "password": args.password or secrets.token_urlsafe(12),
        "gpu": gpu_info,
    }
    state_path = os.path.join(working, STATE_FILE)
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    say(f"state written to {state_path}")
    print(f"\n{BOLD}准备完成。启动服务：{RESET}")
    print(f"  python {os.path.join(ROOT, 'runtime', 'run_all.py')} --state {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
