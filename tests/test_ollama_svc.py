"""Tests for the ollama service helpers against a fake ollama HTTP API.

`ollama_svc` is the piece most likely to waste a whole session: if `ensure_model`
decides a model is already present when it is not, the translation stage fails
20 minutes later with a 404 from the LLM. So the presence check, the fallback
chain and the OpenAI-route verification all get real coverage here, with a stub
server standing in for ollama.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load():
    path = os.path.join(ROOT, "runtime", "ollama_svc.py")
    spec = importlib.util.spec_from_file_location("asmrdub_ollama_svc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def svc():
    return _load()


class _FakeOllama(BaseHTTPRequestHandler):
    models: list[str] = []
    reply: str = "你好。"
    status: int = 200

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/tags"):
            self._json(200, {"models": [{"name": name} for name in type(self).models]})
        else:
            self._json(404, {"error": "nope"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if type(self).status != 200:
            self._json(type(self).status, {"error": "unsupported architecture"})
            return
        self._json(
            200,
            {
                "choices": [
                    {"message": {"role": "assistant", "content": type(self).reply}}
                ]
            },
        )

    def _json(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        """Quiet."""


@pytest.fixture
def fake_ollama():
    _FakeOllama.models = []
    _FakeOllama.reply = "你好。"
    _FakeOllama.status = 200
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1], _FakeOllama
    httpd.shutdown()
    httpd.server_close()


# --------------------------------------------------------------------------
# Presence detection
# --------------------------------------------------------------------------


def test_exact_tag_counts_as_present(svc):
    assert svc._has_model(
        "hf.co/zaakirio/gemma-4-12b-it-uncensored-GGUF:Q4_K_M",
        ["hf.co/zaakirio/gemma-4-12b-it-uncensored-GGUF:Q4_K_M"],
    )


def test_same_repo_different_quant_counts_as_present(svc):
    """Any quant of the right model works; re-pulling 7.4GB does not."""
    assert svc._has_model(
        "hf.co/zaakirio/gemma-4-12b-it-uncensored-GGUF:Q4_K_M",
        ["hf.co/zaakirio/gemma-4-12b-it-uncensored-GGUF:Q5_K_M"],
    )


def test_a_different_model_is_not_a_match(svc):
    """The old substring check matched this and skipped the pull."""
    assert not svc._has_model("gemma4:12b", ["gemma4-something-else:latest"])
    assert not svc._has_model("gemma4:12b", ["llama3:8b", "qwen2.5:7b"])


def test_empty_model_list_is_not_a_match(svc):
    assert not svc._has_model("gemma4:12b", [])


def test_case_is_ignored(svc):
    assert svc._has_model("HF.CO/Owner/Repo:Q4_K_M", ["hf.co/owner/repo:q4_k_m"])


# --------------------------------------------------------------------------
# Model listing and verification against the fake server
# --------------------------------------------------------------------------


def test_list_models_reads_the_tags_endpoint(svc, fake_ollama):
    port, handler = fake_ollama
    handler.models = ["gemma4:12b", "llama3:8b"]
    assert svc.list_models(port) == ["gemma4:12b", "llama3:8b"]


def test_list_models_returns_empty_when_unreachable(svc):
    assert svc.list_models(1) == []


def test_verify_uses_the_openai_route(svc, fake_ollama):
    """This must be the same path ask_gpt takes, or verification proves nothing."""
    port, handler = fake_ollama
    handler.reply = "你好。"
    assert svc.verify("gemma4:12b", port=port, timeout=10) is True


def test_verify_fails_on_an_http_error(svc, fake_ollama):
    """An unsupported GGUF architecture 500s here rather than mid-translation."""
    port, handler = fake_ollama
    handler.status = 500
    assert svc.verify("gemma4:12b", port=port, timeout=10) is False


def test_verify_fails_on_an_empty_completion(svc, fake_ollama):
    port, handler = fake_ollama
    handler.reply = "   "
    assert svc.verify("gemma4:12b", port=port, timeout=10) is False


def test_verify_fails_when_nothing_is_listening(svc):
    assert svc.verify("gemma4:12b", port=1, timeout=2) is False


# --------------------------------------------------------------------------
# ensure_model decision logic
# --------------------------------------------------------------------------


def test_present_model_is_not_re_pulled(svc, fake_ollama, monkeypatch):
    port, handler = fake_ollama
    handler.models = ["hf.co/zaakirio/gemma-4-12b-it-uncensored-GGUF:Q4_K_M"]
    pulls = []
    monkeypatch.setattr(svc, "pull", lambda *a, **k: pulls.append(a) or True)

    chosen = svc.ensure_model("ollama", "/tmp/models", port=port)
    assert chosen == svc.pins.OLLAMA_MODEL
    assert pulls == [], "a present model must not be pulled again"


def test_falls_back_when_the_preferred_pull_fails(svc, fake_ollama, monkeypatch):
    port, _ = fake_ollama
    attempted = []

    def fake_pull(binary, model, models_dir, gpu=None, port_=None, **kwargs):
        attempted.append(model)
        return model != svc.pins.OLLAMA_MODEL   # preferred fails, fallback works

    monkeypatch.setattr(svc, "pull", fake_pull)
    chosen = svc.ensure_model("ollama", "/tmp/models", port=port)
    assert chosen == svc.pins.OLLAMA_MODEL_FALLBACK
    assert attempted == [svc.pins.OLLAMA_MODEL, svc.pins.OLLAMA_MODEL_FALLBACK]


def test_falls_back_when_the_preferred_model_pulls_but_cannot_run(
    svc, fake_ollama, monkeypatch
):
    """A GGUF ollama cannot execute pulls fine and then 500s on inference."""
    port, handler = fake_ollama
    monkeypatch.setattr(svc, "pull", lambda *a, **k: True)
    calls = []

    def fake_verify(model, port=None, timeout=None):
        calls.append(model)
        return model == svc.pins.OLLAMA_MODEL_FALLBACK

    monkeypatch.setattr(svc, "verify", fake_verify)
    chosen = svc.ensure_model("ollama", "/tmp/models", port=port)
    assert chosen == svc.pins.OLLAMA_MODEL_FALLBACK
    assert calls == [svc.pins.OLLAMA_MODEL, svc.pins.OLLAMA_MODEL_FALLBACK]


def test_raises_a_useful_error_when_both_fail(svc, fake_ollama, monkeypatch):
    port, _ = fake_ollama
    monkeypatch.setattr(svc, "pull", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="internet"):
        svc.ensure_model("ollama", "/tmp/models", port=port)


def test_pull_retries_once_on_a_transient_failure(svc, monkeypatch):
    """A 7.4GB download over a flaky link deserves one resume attempt."""
    attempts = []

    class Result:
        def __init__(self, code):
            self.returncode = code

    def fake_run(command, env=None):
        attempts.append(command)
        return Result(1 if len(attempts) == 1 else 0)

    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    assert svc.pull("ollama", "some:model", "/tmp/models") is True
    assert len(attempts) == 2


def test_pull_gives_up_after_two_attempts(svc, monkeypatch):
    attempts = []

    class Result:
        returncode = 1

    def fake_run(command, env=None):
        attempts.append(command)
        return Result()

    monkeypatch.setattr(svc.subprocess, "run", fake_run)
    assert svc.pull("ollama", "some:model", "/tmp/models") is False
    assert len(attempts) == 2


# --------------------------------------------------------------------------
# Server environment
# --------------------------------------------------------------------------


def test_serve_env_pins_the_llm_to_its_own_gpu(svc, monkeypatch, tmp_path):
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs.get("env", {})
            self.pid = 4242

    monkeypatch.setattr(svc.subprocess, "Popen", FakePopen)
    svc.start_server("ollama", str(tmp_path / "models"), str(tmp_path / "log.txt"))

    env = captured["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == svc.pins.LLM_GPU
    assert env["CUDA_VISIBLE_DEVICES"] != svc.pins.WORKER_GPU
    # Loopback only: a quick-tunnel URL is public and ollama has no auth.
    assert env["OLLAMA_HOST"].startswith("127.0.0.1:")
    # Weights must not land in the 20GB working quota.
    assert env["OLLAMA_MODELS"] == str(tmp_path / "models")
    # Turing has no flash attention.
    assert env["OLLAMA_FLASH_ATTENTION"] == "0"
    assert captured["command"] == ["ollama", "serve"]


def test_serve_keeps_the_model_resident(svc, monkeypatch, tmp_path):
    """A 7.4GB reload between chunks would dominate the translation stage."""
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["env"] = kwargs.get("env", {})
            self.pid = 1

    monkeypatch.setattr(svc.subprocess, "Popen", FakePopen)
    svc.start_server("ollama", str(tmp_path / "m"), str(tmp_path / "l.txt"))
    assert captured["env"]["OLLAMA_KEEP_ALIVE"] == "2h"
