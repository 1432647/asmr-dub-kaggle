"""Start and feed the local translation LLM (ollama on GPU 1).

Why ollama and not llama.cpp directly: llama.cpp's GitHub releases have no
Linux CUDA build (only CPU / vulkan / sycl / rocm), so it would mean a 20-30
minute source compile every session. Why not vLLM: it installs its own torch
and would collide with the worker's environment. Ollama is a single tarball
with a bundled CUDA runtime, zero Python dependencies, and speaks the
OpenAI-compatible `/v1` API that VideoLingo's ask_gpt requires.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asmrdub import pins

BOLD = "\033[1m"
RESET = "\033[0m"


def say(message: str) -> None:
    print(f"{BOLD}[llm]{RESET} {message}", flush=True)


def base_url(port: int = pins.OLLAMA_PORT) -> str:
    return f"http://127.0.0.1:{port}"


def start_server(
    binary: str,
    models_dir: str,
    log_path: str,
    gpu: str = pins.LLM_GPU,
    port: int = pins.OLLAMA_PORT,
    parallel: int = 2,
    context: int = 8192,
) -> subprocess.Popen:
    """Launch `ollama serve` pinned to one GPU."""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    env = {
        **os.environ,
        # The translator gets its own card so it never competes with TTS.
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        # Weights live in scratch: /kaggle/working has a hard 20GB cap.
        "OLLAMA_MODELS": models_dir,
        # VideoLingo translates chunks concurrently; 2 contexts fit next to a
        # 7.4GB Q4_K_M on a 15GB T4.
        "OLLAMA_NUM_PARALLEL": str(parallel),
        # The translation prompts include summary + terminology + context lines.
        "OLLAMA_CONTEXT_LENGTH": str(context),
        # Reloading a 7.4GB model between chunks would dominate the runtime.
        "OLLAMA_KEEP_ALIVE": "2h",
        "OLLAMA_FLASH_ATTENTION": "0",  # not supported on Turing
    }
    log = open(log_path, "ab")
    process = subprocess.Popen(
        [binary, "serve"], stdout=log, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )
    say(f"ollama serve started (pid={process.pid}, gpu={gpu}, port={port})")
    return process


def wait_ready(port: int = pins.OLLAMA_PORT, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url(port)}/api/tags", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - "not up yet" is expected
            time.sleep(2)
    return False


def list_models(port: int = pins.OLLAMA_PORT) -> list[str]:
    try:
        with urllib.request.urlopen(f"{base_url(port)}/api/tags", timeout=10) as resp:
            payload = json.load(resp)
    except Exception:  # noqa: BLE001
        return []
    return [entry.get("name", "") for entry in payload.get("models", [])]


def pull(binary: str, model: str, models_dir: str, gpu: str = pins.LLM_GPU,
         port: int = pins.OLLAMA_PORT) -> bool:
    """Pull a model, streaming progress. False on failure rather than raising.

    A failed pull is recoverable (we fall back to the official tag), so the
    caller decides what to do. `ollama pull` is retried once: the 7.4GB download
    fails on a transient network error often enough to be worth it, and it
    resumes rather than restarting.
    """
    env = {
        **os.environ,
        "OLLAMA_HOST": f"127.0.0.1:{port}",
        "OLLAMA_MODELS": models_dir,
        "CUDA_VISIBLE_DEVICES": str(gpu),
    }
    for attempt in (1, 2):
        say(f"pulling {model} (this is the 7.4GB download, attempt {attempt}/2)")
        result = subprocess.run([binary, "pull", model], env=env)
        if result.returncode == 0:
            return True
        say(f"pull failed for {model} (rc={result.returncode})")
    return False


def _has_model(name: str, existing: list[str]) -> bool:
    """Is `name` already pulled?

    ollama reports names with a tag appended (`gemma4:12b`, and for HF pulls the
    full `hf.co/owner/repo:QUANT`). Compare on the whole string and on the
    untagged repo, but never on a bare substring: `gemma4` would otherwise match
    `gemma4-something-else` and skip the pull we actually need.
    """
    wanted = name.lower()
    wanted_repo = wanted.split(":", 1)[0]
    for entry in existing:
        entry = entry.lower()
        if entry == wanted or entry.split(":", 1)[0] == wanted_repo:
            return True
    return False


def ensure_model(
    binary: str,
    models_dir: str,
    preferred: str = pins.OLLAMA_MODEL,
    fallback: str = pins.OLLAMA_MODEL_FALLBACK,
    gpu: str = pins.LLM_GPU,
    port: int = pins.OLLAMA_PORT,
) -> str:
    """Return a usable model tag, falling back if the preferred one fails.

    The preferred tag is a third-party de-censored Gemma 4 12B GGUF. ASMR
    content makes refusals a real failure mode: ask_gpt retries five times and
    then aborts the stage, so a model that declines is worse than a weaker one.
    The official `gemma4:12b` is the fallback -- it will run, but may refuse.
    """
    existing = list_models(port)
    for candidate in (preferred, fallback):
        if _has_model(candidate, existing):
            say(f"model already present: {candidate}")
            return candidate

    if pull(binary, preferred, models_dir, gpu, port) and verify(preferred, port):
        return preferred

    say("falling back to the official (censored) gemma4:12b")
    if pull(binary, fallback, models_dir, gpu, port) and verify(fallback, port):
        say(
            "WARNING using the official model: it may refuse to translate "
            "explicit material, which aborts the translation stage."
        )
        return fallback
    raise RuntimeError(
        "no translation model could be pulled. Check notebook internet access "
        "(Settings -> Internet must be On, which requires phone verification)."
    )


def verify(model: str, port: int = pins.OLLAMA_PORT, timeout: int = 900) -> bool:
    """One real completion through the OpenAI-compatible route.

    This is the exact path ask_gpt uses, so a success here means translation
    will work; a 500 from an unsupported GGUF architecture shows up now rather
    than 20 minutes in.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "把这句日语翻译成中文：こんにちは。只输出译文。"}],
            "temperature": 0.0,
            "max_tokens": 64,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url(port)}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
        text = body["choices"][0]["message"]["content"].strip()
        say(f"verify ok: {text[:60]!r}")
        return bool(text)
    except urllib.error.HTTPError as exc:
        say(f"verify failed: HTTP {exc.code} {exc.read()[:300]!r}")
        return False
    except Exception as exc:  # noqa: BLE001
        say(f"verify failed: {exc}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--model", default=pins.OLLAMA_MODEL)
    args = parser.parse_args()

    start_server(args.binary, args.models_dir, args.log)
    if not wait_ready():
        raise SystemExit("ollama did not become ready; see " + args.log)
    print(ensure_model(args.binary, args.models_dir, preferred=args.model))
