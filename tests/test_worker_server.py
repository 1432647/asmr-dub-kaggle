"""Tests for the GPU worker's HTTP layer and its ASR/TTS contracts.

`worker/server.py` imports torch only inside functions, so the routing, error
reporting and serialisation can all be tested here without CUDA. The handlers
themselves are monkeypatched; what is under test is everything around them --
which is where a wedged pipeline usually comes from (a 500 with no traceback,
a route typo, two requests racing onto one GPU).

Also checks `asmrdub.asr_format` against the real faster-whisper dataclasses
rather than stand-ins, so a field rename upstream shows up here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_worker():
    """Import worker/server.py by path -- `worker` is not an installed package."""
    path = os.path.join(ROOT, "worker", "server.py")
    spec = importlib.util.spec_from_file_location("asmrdub_worker_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def worker():
    return _load_worker()


@pytest.fixture
def server(worker):
    """A live worker HTTP server with the GPU handlers replaced."""
    calls: list[tuple[str, dict]] = []

    def fake_tts(payload):
        calls.append(("tts", payload))
        return {"ok": True, "out": payload.get("out"), "seconds": 0.5}

    def fake_asr(payload):
        calls.append(("asr", payload))
        return {"ok": True, "result": {"segments": [], "language": "ja"}}

    def fake_separate(payload):
        calls.append(("separate", payload))
        return {"ok": True, "duration": 12.0, "chunks": 3}

    def boom(payload):
        raise RuntimeError("CUDA out of memory (simulated)")

    original = dict(worker.ROUTES)
    worker.ROUTES.clear()
    worker.ROUTES.update(
        {"/tts": fake_tts, "/asr": fake_asr, "/separate": fake_separate,
         "/boom": boom}
    )
    worker.STATE["log"] = None
    worker.STATE["loaded"] = None
    worker.STATE["device"] = "cuda:0"

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), worker.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, calls
    httpd.shutdown()
    httpd.server_close()
    worker.ROUTES.clear()
    worker.ROUTES.update(original)


def post(base: str, route: str, payload: dict, timeout: int = 10):
    request = urllib.request.Request(
        base + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


# --------------------------------------------------------------------------
# Routing and error handling
# --------------------------------------------------------------------------


def test_health_reports_state(server):
    base, _ = server
    with urllib.request.urlopen(base + "/health", timeout=10) as response:
        body = json.loads(response.read())
    assert body["ok"] is True
    assert "loaded" in body and "device" in body


def test_routes_dispatch_by_path(server):
    base, calls = server
    post(base, "/tts", {"text": "x", "out": "/tmp/x.wav"})
    post(base, "/asr", {"input": "/tmp/a.wav"})
    post(base, "/separate", {"input": "/tmp/a.wav"})
    assert [name for name, _ in calls] == ["tts", "asr", "separate"]


def test_trailing_slash_is_accepted(server):
    base, calls = server
    post(base, "/tts/", {"text": "x"})
    assert calls and calls[-1][0] == "tts"


def test_unknown_route_is_404_not_a_hang(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as info:
        post(base, "/nope", {})
    assert info.value.code == 404


def test_malformed_json_is_400(server):
    base, _ = server
    request = urllib.request.Request(
        base + "/tts", data=b"{not json", headers={"Content-Type": "application/json"}
    )
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(request, timeout=10)
    assert info.value.code == 400


def test_handler_exception_returns_traceback(server):
    """Debugging an OOM from a bare 500 is miserable; the trace must come back."""
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as info:
        post(base, "/boom", {})
    assert info.value.code == 500
    body = json.loads(info.value.read())
    assert body["ok"] is False
    assert "CUDA out of memory" in body["error"]
    assert "traceback" in body and "RuntimeError" in body["traceback"]


def test_worker_survives_a_failed_request(server):
    """One bad request must not take the process down mid-run."""
    base, calls = server
    with pytest.raises(urllib.error.HTTPError):
        post(base, "/boom", {})
    status, body = post(base, "/tts", {"text": "still alive"})
    assert status == 200 and body["ok"] is True
    assert calls[-1][1]["text"] == "still alive"


def test_requests_are_serialised(server, worker):
    """Two concurrent calls must not run at once: there is one GPU."""
    overlap = []
    active = {"count": 0}
    lock = threading.Lock()

    def slow(payload):
        with lock:
            active["count"] += 1
            overlap.append(active["count"])
        time.sleep(0.3)
        with lock:
            active["count"] -= 1
        return {"ok": True}

    worker.ROUTES["/slow"] = slow
    base, _ = server
    threads = [
        threading.Thread(target=lambda: post(base, "/slow", {}, timeout=20))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(overlap) == 1, f"handlers overlapped: {overlap}"


# --------------------------------------------------------------------------
# Client behaviour
# --------------------------------------------------------------------------


def test_client_reports_worker_errors_with_detail(server, monkeypatch):
    """The app-side client must surface the worker's traceback, not swallow it."""
    base, _ = server
    client_path = os.path.join(
        ROOT, "overlay", "core", "asr_backend", "worker_client.py"
    )
    spec = importlib.util.spec_from_file_location("asmrdub_worker_client", client_path)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)

    monkeypatch.setenv("ASMRDUB_WORKER", base)
    assert client.health() is not None
    with pytest.raises(client.WorkerError) as info:
        client.call("/boom", {})
    assert "CUDA out of memory" in str(info.value)
    assert "RuntimeError" in str(info.value)


def test_client_reports_unreachable_worker(monkeypatch):
    client_path = os.path.join(
        ROOT, "overlay", "core", "asr_backend", "worker_client.py"
    )
    spec = importlib.util.spec_from_file_location("asmrdub_worker_client2", client_path)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    # Port 1 is reserved and never listening.
    monkeypatch.setenv("ASMRDUB_WORKER", "http://127.0.0.1:1")
    assert client.health(timeout=2) is None
    with pytest.raises(client.WorkerError, match="cannot reach worker"):
        client.call("/tts", {}, timeout=2)


def test_client_injects_the_package_path(server, monkeypatch):
    """The worker sys.path-inserts this to import asmrdub from its own venv."""
    base, calls = server
    client_path = os.path.join(
        ROOT, "overlay", "core", "asr_backend", "worker_client.py"
    )
    spec = importlib.util.spec_from_file_location("asmrdub_worker_client3", client_path)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    monkeypatch.setenv("ASMRDUB_WORKER", base)
    monkeypatch.setenv("ASMRDUB_PKG_PATH", "/opt/asmrdub")
    client.call("/tts", {"text": "x"})
    assert calls[-1][1]["asmrdub_path"] == "/opt/asmrdub"


# --------------------------------------------------------------------------
# ASR format against the real faster-whisper types
# --------------------------------------------------------------------------


def test_format_handles_real_faster_whisper_dataclasses():
    """Guards against a field rename in faster-whisper's Segment/Word."""
    transcribe = pytest.importorskip("faster_whisper.transcribe")
    from asmrdub.asr_format import to_videolingo

    word = transcribe.Word(start=1.0, end=1.6, word="こんばんは", probability=0.9)
    segment = transcribe.Segment(
        id=1, seek=0, start=1.0, end=1.6, text="こんばんは",
        tokens=[1, 2, 3], avg_logprob=-0.2, compression_ratio=1.1,
        no_speech_prob=0.01, words=[word], temperature=0.0,
    )
    result = to_videolingo([segment], "ja")
    assert result["segments"][0]["text"] == "こんばんは"
    assert result["segments"][0]["words"][0]["word"] == "こんばんは"
    assert result["segments"][0]["words"][0]["start"] == 1.0


def test_format_handles_a_segment_with_no_words():
    transcribe = pytest.importorskip("faster_whisper.transcribe")
    from asmrdub.asr_format import to_videolingo

    segment = transcribe.Segment(
        id=1, seek=0, start=2.0, end=2.5, text="ふぅ", tokens=[1],
        avg_logprob=-0.3, compression_ratio=1.0, no_speech_prob=0.2,
        words=None, temperature=0.0,
    )
    result = to_videolingo([segment], "ja")
    assert result["segments"][0]["words"] == [
        {"word": "ふぅ", "start": 2.0, "end": 2.5}
    ]


def test_worker_vad_parameters_are_all_real_options():
    """A typo'd VAD key is silently ignored or raises deep inside the model."""
    vad = pytest.importorskip("faster_whisper.vad")
    valid = set(vad.VadOptions.__annotations__)
    used = {
        "threshold",
        "min_speech_duration_ms",
        "min_silence_duration_ms",
        "speech_pad_ms",
    }
    assert used <= valid, f"unknown VAD options: {used - valid}"


def test_worker_transcribe_kwargs_exist():
    """Every kwarg the worker passes must be accepted by this version."""
    import inspect

    model = pytest.importorskip("faster_whisper").WhisperModel
    accepted = set(inspect.signature(model.transcribe).parameters)
    used = {
        "language", "task", "beam_size", "word_timestamps", "vad_filter",
        "vad_parameters", "temperature", "condition_on_previous_text",
        "no_speech_threshold",
    }
    assert used <= accepted, f"unsupported kwargs: {used - accepted}"


def test_worker_model_init_kwargs_exist():
    import inspect

    model = pytest.importorskip("faster_whisper").WhisperModel
    accepted = set(inspect.signature(model.__init__).parameters)
    assert {"device", "device_index", "compute_type"} <= accepted
