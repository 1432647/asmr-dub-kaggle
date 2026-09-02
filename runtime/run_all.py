"""Start every service, publish the tunnel, then keep the session alive.

Startup order is deliberate:

1. ollama (GPU 1) -- the 7.4GB pull is the longest single step, so it starts
   first and downloads while the rest comes up.
2. GPU worker (GPU 0) -- loads nothing until the first request, so "ready"
   means the port answers.
3. Streamlit UI -- the only thing published.
4. cloudflared -- once the UI answers, so the first visitor is not greeted by
   a 502.

Then a keepalive loop watches all three and restarts the tunnel if it dies.
Killing this cell leaves the services running (they are all in their own
session), so the notebook can be re-run without a restart.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from asmrdub import pins  # noqa: E402
from runtime import ollama_svc, tunnel  # noqa: E402

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def say(message: str) -> None:
    print(f"{BOLD}[run]{RESET} {message}", flush=True)


def banner(message: str) -> None:
    print(f"\n{GREEN}{'=' * 68}\n{message}\n{'=' * 68}{RESET}\n", flush=True)


def http_ok(url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except Exception:  # noqa: BLE001
        return False


def wait_port(url: str, timeout: int, label: str) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if http_ok(url):
            say(f"{label} ready")
            return True
        time.sleep(2)
    say(f"{label} did NOT become ready within {timeout}s")
    return False


def start_worker(state: dict, log_path: str) -> subprocess.Popen:
    command = [
        state["gpu_python"],
        os.path.join(ROOT, "worker", "server.py"),
        "--port", str(pins.WORKER_PORT),
        "--index-tts-root", state["index_tts_root"],
        "--model-dir", state["models"]["indextts"],
        "--whisper-dir", state["models"]["whisper"],
        "--hdemucs", state["models"]["hdemucs"],
        "--gpu", pins.WORKER_GPU,
        "--log", log_path,
    ]
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "ab")
    # PYTHONPATH so the worker can import asmrdub from its own environment.
    env = {**os.environ, "PYTHONPATH": ROOT}
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )
    say(f"worker started (pid={process.pid}, gpu={pins.WORKER_GPU})")
    return process


def start_ui(state: dict, password: str, log_path: str) -> subprocess.Popen:
    env = {
        **os.environ,
        "ASMRDUB_PASSWORD": password,
        "ASMRDUB_WORKER": f"http://127.0.0.1:{pins.WORKER_PORT}",
        "ASMRDUB_PKG_PATH": ROOT,
        "ASMRDUB_INPUT_ROOT": state["input_root"],
        "ASMRDUB_LOG": os.path.join(state["repo_root"], "output", "pipeline.log"),
        # Streamlit's usage stats ping adds a startup stall behind a firewall.
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "PYTHONPATH": ROOT,
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        state["app_python"], "-m", "streamlit", "run", "asmr_ui.py",
        "--server.port", str(pins.UI_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        # A 15-minute WAV can exceed the 200MB default when uploading directly.
        "--server.maxUploadSize", "2048",
        "--server.enableXsrfProtection", "false",
        "--browser.gatherUsageStats", "false",
    ]
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "ab")
    process = subprocess.Popen(
        command, cwd=state["repo_root"], stdout=log, stderr=subprocess.STDOUT,
        env=env, start_new_session=True,
    )
    say(f"streamlit started (pid={process.pid}, port={pins.UI_PORT})")
    return process


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, help="state json from bootstrap")
    parser.add_argument("--no-keepalive", action="store_true")
    args = parser.parse_args()

    with open(args.state, "r", encoding="utf-8") as fh:
        state = json.load(fh)

    logs = os.path.join(state["scratch"], "logs")
    os.makedirs(logs, exist_ok=True)
    password = state.get("password") or secrets.token_urlsafe(12)

    # --- 1. LLM (starts first: the model pull is the long pole) -------------
    ollama_log = os.path.join(logs, "ollama.log")
    ollama_svc.start_server(
        state["ollama_binary"],
        os.path.join(state["scratch"], "ollama-models"),
        ollama_log,
    )
    if not ollama_svc.wait_ready():
        print(open(ollama_log, errors="ignore").read()[-2000:])
        raise SystemExit("ollama failed to start; see the log above")
    model = ollama_svc.ensure_model(
        state["ollama_binary"], os.path.join(state["scratch"], "ollama-models")
    )
    say(f"translation model: {model}")
    if model != state.get("ollama_model"):
        _rewrite_model(state, model)

    # --- 2. GPU worker -----------------------------------------------------
    worker_log = os.path.join(logs, "worker.log")
    start_worker(state, worker_log)
    if not wait_port(
        f"http://127.0.0.1:{pins.WORKER_PORT}/health", 180, "gpu worker"
    ):
        print(open(worker_log, errors="ignore").read()[-3000:])
        raise SystemExit("gpu worker failed to start; see the log above")

    # --- 3. UI -------------------------------------------------------------
    ui_log = os.path.join(logs, "streamlit.log")
    start_ui(state, password, ui_log)
    if not wait_port(f"http://127.0.0.1:{pins.UI_PORT}", 180, "streamlit"):
        print(open(ui_log, errors="ignore").read()[-3000:])
        raise SystemExit("streamlit failed to start; see the log above")

    # --- 4. Tunnel ---------------------------------------------------------
    tunnel_log = os.path.join(logs, "tunnel.log")
    tunnel.start(state["cloudflared_binary"], pins.UI_PORT, tunnel_log)
    url = tunnel.wait_url(tunnel_log)
    if not url:
        print(tunnel.tail(tunnel_log))
        raise SystemExit("no tunnel URL appeared; see the log above")

    banner(
        f"网页地址: {url}\n"
        f"访问密码: {password}\n\n"
        f"翻译模型: {model}\n"
        f"日志:     {logs}/\n\n"
        f"{YELLOW}这个地址是公开的，只有密码保护。用完请停止会话。{RESET}"
    )

    if args.no_keepalive:
        return 0

    say("keepalive running — Ctrl-C 停止本 cell（服务会继续运行）")
    while True:
        ui_alive = http_ok(f"http://127.0.0.1:{pins.UI_PORT}")
        worker_alive = http_ok(f"http://127.0.0.1:{pins.WORKER_PORT}/health")
        llm_alive = http_ok(f"http://127.0.0.1:{pins.OLLAMA_PORT}/api/tags")
        tunnel_alive = http_ok(url, timeout=8)
        print(
            "%s ui=%s worker=%s llm=%s tunnel=%s"
            % (
                time.strftime("%H:%M:%S"),
                ui_alive, worker_alive, llm_alive, tunnel_alive,
            ),
            flush=True,
        )
        if ui_alive and not tunnel_alive:
            say("tunnel looks dead, restarting cloudflared")
            subprocess.run("pkill -f cloudflared || true", shell=True)
            time.sleep(2)
            tunnel.start(state["cloudflared_binary"], pins.UI_PORT, tunnel_log)
            new_url = tunnel.wait_url(tunnel_log, timeout=90)
            if new_url:
                url = new_url
                banner(f"新的网页地址: {url}\n访问密码: {password}")
        if not worker_alive:
            say("WARNING gpu worker is down; see logs/worker.log")
        time.sleep(60)


def _rewrite_model(state: dict, model: str) -> None:
    """Point VideoLingo's config at the model we actually ended up with."""
    repo_root = state["repo_root"]
    sys.path.insert(0, repo_root)
    previous = os.getcwd()
    os.chdir(repo_root)
    try:
        from core.utils.config_utils import update_key

        update_key("api.model", model)
        say(f"config api.model = {model}")
    except Exception as exc:  # noqa: BLE001
        say(f"WARNING could not update api.model: {exc}")
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    raise SystemExit(main())
