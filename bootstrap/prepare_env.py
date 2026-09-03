"""Build the two virtualenvs and fetch the standalone binaries.

Two environments, because VideoLingo and IndexTTS have incompatible pins
(librosa 0.11 vs 0.10.2, opencv 4.11 vs 4.9, transformers latest vs 4.52.1):

* app  -- CPU only, runs Streamlit + orchestration. Never imports torch.
* gpu  -- index-tts's own uv project, plus faster-whisper. All CUDA work.

Neither touches Kaggle's preinstalled site-packages: `uv venv` without
--system-site-packages means a numpy upgrade here cannot break the notebook's
own environment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asmrdub import pins
from bootstrap import zst

BOLD = "\033[1m"
RESET = "\033[0m"


def say(message: str) -> None:
    print(f"{BOLD}[env]{RESET} {message}", flush=True)


def run(command: list[str], cwd: str | None = None, env: dict | None = None,
        retries: int = 0, what: str = "") -> None:
    """Run a command, retrying transient failures.

    Package installs pull gigabytes over flaky networks; a bare failure here
    would waste the whole session, so retry before giving up.
    """
    for attempt in range(retries + 1):
        say("$ " + " ".join(command))
        merged = {**os.environ, **(env or {})}
        result = subprocess.run(command, cwd=cwd, env=merged)
        if result.returncode == 0:
            return
        if attempt < retries:
            say(f"failed (rc={result.returncode}), retrying {attempt + 1}/{retries}")
        else:
            raise RuntimeError(
                f"command failed{f' ({what})' if what else ''}: {' '.join(command)}"
            )


def ensure_uv() -> str:
    """Path to a uv executable, installing it with pip when absent."""
    found = shutil.which("uv")
    if found:
        return found
    say("installing uv")
    run([sys.executable, "-m", "pip", "install", "-q", "uv"], retries=2, what="uv")
    found = shutil.which("uv")
    if found:
        return found
    # pip --user installs land outside PATH in some images.
    for candidate in (
        os.path.expanduser("~/.local/bin/uv"),
        "/usr/local/bin/uv",
        os.path.join(os.path.dirname(sys.executable), "uv"),
    ):
        if os.path.isfile(candidate):
            return candidate
    raise RuntimeError("uv installed but not found on PATH")


def venv_python(venv_dir: str) -> str:
    return os.path.join(
        venv_dir, "Scripts" if os.name == "nt" else "bin",
        "python.exe" if os.name == "nt" else "python",
    )


# --------------------------------------------------------------------------
# App environment (CPU)
# --------------------------------------------------------------------------


def build_app_env(uv: str, venv_dir: str, repo_root: str) -> str:
    requirements = os.path.join(repo_root, "requirements-app.txt")
    if not os.path.isfile(requirements):
        raise RuntimeError("requirements-app.txt missing -- run apply_overlay first")

    python = venv_python(venv_dir)
    if not os.path.isfile(python):
        run([uv, "venv", venv_dir, "--python", "3.11"], what="app venv")

    run(
        [uv, "pip", "install", "--python", python, "-r", requirements],
        retries=2,
        what="app requirements",
    )
    # spaCy's Japanese pipeline. Installed from the direct wheel URL because
    # `spacy download` needs a working `spacy` CLI inside this venv and one
    # extra subprocess layer to fail in.
    run(
        [
            uv, "pip", "install", "--python", python,
            "https://github.com/explosion/spacy-models/releases/download/"
            "ja_core_news_md-3.8.0/ja_core_news_md-3.8.0-py3-none-any.whl",
        ],
        retries=2,
        what="spaCy ja model",
    )
    say("app environment ready")
    return python


# --------------------------------------------------------------------------
# GPU environment (index-tts's own uv project)
# --------------------------------------------------------------------------


def build_gpu_env(uv: str, index_tts_root: str) -> str:
    """`uv sync` the index-tts project, then add faster-whisper.

    Deliberately NOT `--all-extras`: that would pull flash-attn, which does not
    support Turing (T4 = sm_75) and fails at the first kernel call, and
    deepspeed, which compiles for minutes to no benefit at batch size 1.
    """
    env = {
        # Keep the venv inside the checkout so the worker's --index-tts-root is
        # the only path anything needs to know.
        "UV_PROJECT_ENVIRONMENT": os.path.join(index_tts_root, ".venv"),
        "UV_HTTP_TIMEOUT": "300",
    }
    run([uv, "sync"], cwd=index_tts_root, env=env, retries=2, what="index-tts sync")
    python = venv_python(os.path.join(index_tts_root, ".venv"))
    run(
        [uv, "pip", "install", "--python", python, "faster-whisper==1.2.1"],
        retries=2,
        what="faster-whisper",
    )
    say("gpu environment ready")
    return python


def verify_gpu_env(python: str) -> dict:
    """Confirm torch sees both T4s and that the ASR backend imports.

    Failing here costs 30 seconds; failing at first inference costs the ~15
    minutes already spent on ASR and separation.
    """
    probe = (
        "import json, torch, faster_whisper;"
        "print(json.dumps({"
        "'torch': torch.__version__,"
        "'cuda': torch.cuda.is_available(),"
        "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
        "'capability': [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())],"
        "}))"
    )
    result = subprocess.run([python, "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gpu env verification failed:\n{result.stderr[-1500:]}")
    import json

    info = json.loads(result.stdout.strip().splitlines()[-1])
    say(f"torch {info['torch']} · cuda={info['cuda']} · devices={info['devices']}")
    if not info["cuda"]:
        raise RuntimeError(
            "torch cannot see a GPU. In the notebook sidebar set Accelerator to "
            "'GPU T4 x2' and restart the session."
        )
    if len(info["devices"]) < 2:
        say(
            "WARNING only %d GPU visible; the translator will share a card with "
            "TTS and may run out of memory" % len(info["devices"])
        )
    return info


# --------------------------------------------------------------------------
# Standalone binaries
# --------------------------------------------------------------------------


def install_ollama(
    scratch: str, helper_python: str | None = None, uv: str | None = None
) -> str:
    """Download and extract ollama; return the path to the binary.

    Only `.tar.zst` is published now (the `.tgz` URL 404s), and Kaggle's image
    has no zstd at all -- not the binary, and not the Python module. Upstream's
    own install.sh simply errors out here telling you to apt-get it. `zst.extract`
    tries a binary, an in-process decompressor, a helper interpreter and finally
    apt, so a missing zstd stops being fatal.

    Downloads to a `.part` file and renames on success: a truncated 1.4GB archive
    left behind by a dropped connection would be skipped as "already downloaded"
    on the next run and then fail to extract, which is a confusing way to lose a
    session.
    """
    root = os.path.join(scratch, "ollama")
    binary = os.path.join(root, "bin", "ollama")
    if os.path.isfile(binary):
        say("ollama already installed")
        return binary
    os.makedirs(root, exist_ok=True)

    archive = os.path.join(scratch, "ollama-linux-amd64.tar.zst")
    if not os.path.isfile(archive):
        say("downloading ollama (~1.4GB)")
        partial = archive + ".part"
        with urllib.request.urlopen(pins.OLLAMA_TARBALL, timeout=300) as response, open(
            partial, "wb"
        ) as fh:
            shutil.copyfileobj(response, fh, length=1 << 20)
        expected = pins.OLLAMA_TARBALL_SIZE
        actual = os.path.getsize(partial)
        if expected and actual != expected:
            os.unlink(partial)
            raise RuntimeError(
                f"ollama archive is {actual} bytes, expected {expected} "
                f"(download truncated; re-run to retry)"
            )
        os.replace(partial, archive)

    say(f"extracting ollama (zstd support: {zst.describe_support(helper_python)})")
    method = zst.extract(archive, root, helper_python=helper_python, uv=uv, log=say)
    say(f"extracted via {method}")

    if not os.path.isfile(binary):
        found = _find_named(root, "ollama")
        if not found:
            raise RuntimeError("ollama binary not found after extraction")
        binary = found
    os.chmod(binary, 0o755)
    return binary


def _find_named(root: str, name: str) -> str | None:
    """First file called `name` under `root`, preferring an executable one."""
    fallback = None
    for current, _, files in os.walk(root):
        if name in files:
            candidate = os.path.join(current, name)
            if os.access(candidate, os.X_OK):
                return candidate
            fallback = fallback or candidate
    return fallback


def install_cloudflared(scratch: str) -> str:
    """Fetch the cloudflared binary. Written via `.part` for the same reason.

    A half-written binary is worse than none: it exists, so the next run skips
    the download, and then the tunnel dies with "Exec format error".
    """
    target = os.path.join(scratch, "bin", "cloudflared")
    if os.path.isfile(target) and os.path.getsize(target) > 1_000_000:
        return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    say("downloading cloudflared")
    partial = target + ".part"
    with urllib.request.urlopen(pins.CLOUDFLARED_URL, timeout=180) as response, open(
        partial, "wb"
    ) as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)
    if os.path.getsize(partial) < 1_000_000:
        size = os.path.getsize(partial)
        os.unlink(partial)
        raise RuntimeError(f"cloudflared download is only {size} bytes; retry")
    os.replace(partial, target)
    os.chmod(target, 0o755)
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--index-tts-root", required=True)
    parser.add_argument("--app-venv", required=True)
    parser.add_argument("--scratch", required=True)
    args = parser.parse_args()

    uv = ensure_uv()
    app_python = build_app_env(uv, args.app_venv, args.repo_root)
    gpu_python = build_gpu_env(uv, args.index_tts_root)
    verify_gpu_env(gpu_python)
    install_ollama(args.scratch, helper_python=app_python, uv=uv)
    install_cloudflared(args.scratch)
    say(f"app python: {app_python}")
    say(f"gpu python: {gpu_python}")
