"""HTTP client for the GPU worker (P2).

The app process is deliberately torch-free, so every GPU operation is a JSON
POST to 127.0.0.1. Only the stdlib is used: adding `requests` here would be
harmless but this module is imported by VideoLingo internals during setup, and
keeping it dependency-free means it cannot be the thing that breaks.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE = os.environ.get("ASMRDUB_WORKER", "http://127.0.0.1:7861")

# GPU work is slow and the worker serialises requests, so a queued call can
# legitimately wait a long time. These ceilings only exist to stop a truly
# wedged worker from hanging the UI forever.
TIMEOUT_TTS = 900
TIMEOUT_ASR = 5400
TIMEOUT_SEP = 5400


class WorkerError(RuntimeError):
    """The worker replied with a failure, or could not be reached."""


def worker_base() -> str:
    return os.environ.get("ASMRDUB_WORKER", DEFAULT_BASE).rstrip("/")


def asmrdub_path() -> str:
    """Where the worker can import the pure-logic package from.

    The worker runs in a different virtualenv, so it cannot rely on asmrdub
    being installed; it sys.path-inserts this directory instead.
    """
    return os.environ.get("ASMRDUB_PKG_PATH", "")


def call(route: str, payload: dict, timeout: int = 600) -> dict:
    """POST ``payload`` to ``route`` and return the parsed reply.

    Raises WorkerError on transport failure or a non-ok reply, with the
    worker's traceback attached when it sent one -- debugging a GPU crash from
    a bare 500 is otherwise miserable.
    """
    url = f"{worker_base()}{route}"
    body = dict(payload)
    body.setdefault("asmrdub_path", asmrdub_path())
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        try:
            parsed = json.loads(detail)
            message = parsed.get("error") or detail
            trace = parsed.get("traceback")
        except ValueError:
            message, trace = detail, None
        raise WorkerError(
            f"worker {route} failed: {message}" + (f"\n{trace}" if trace else "")
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise WorkerError(f"cannot reach worker at {url}: {exc}") from exc
    if not reply.get("ok"):
        raise WorkerError(f"worker {route} failed: {reply.get('error')}")
    return reply


def health(timeout: int = 5) -> dict | None:
    """Current worker status, or None when it is not up yet."""
    try:
        with urllib.request.urlopen(f"{worker_base()}/health", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - "not ready" is the common case
        return None


def wait_ready(timeout: int = 300, interval: float = 2.0) -> bool:
    """Block until the worker answers /health, or the timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if health() is not None:
            return True
        time.sleep(interval)
    return False


def unload() -> None:
    """Ask the worker to free VRAM. Best-effort; failure is not fatal."""
    try:
        urllib.request.urlopen(f"{worker_base()}/unload", timeout=60).read()
    except Exception:  # noqa: BLE001
        pass
