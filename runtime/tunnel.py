"""Cloudflare quick tunnel for the Streamlit UI.

Only port 8501 is published. The worker (7861) and ollama (11434) stay on
loopback: they have no authentication at all, and a quick-tunnel URL is public
to anyone who learns it.
"""

from __future__ import annotations

import os
import re
import subprocess
import time

URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def start(binary: str, port: int, log_path: str) -> subprocess.Popen:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    # Truncate: a stale URL from a previous run in the same file would be
    # scraped and reported as the live one.
    log = open(log_path, "wb")
    process = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process


def wait_url(log_path: str, timeout: int = 120) -> str | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                match = URL_PATTERN.search(fh.read())
            if match:
                return match.group(0)
        time.sleep(2)
    return None


def tail(log_path: str, limit: int = 1500) -> str:
    if not os.path.exists(log_path):
        return "(no tunnel log)"
    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()[-limit:]
