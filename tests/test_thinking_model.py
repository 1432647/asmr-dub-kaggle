"""Tests for surviving a thinking model behind an OpenAI-compatible endpoint.

The Kaggle run died here after a successful 7.4GB pull:

    [llm] verify ok: ''
    RuntimeError: no translation model could be pulled.

Gemma 4 is a thinking model. Through ollama's `/v1/chat/completions` the whole
answer lands in a non-standard `reasoning` field and `content` comes back empty
(ollama#15288), so `verify` saw a 200 with no text, called it a failure, and the
fallback chain ran out. Worse, `bool("")` made the log say "ok" while returning
False -- the message actively misled.

Two things are covered:

* `ollama_svc.answer_text` / `verify` against a fake server that reproduces the
  thinking-model shape, plus the non-thinking shape, plus a 200 with no text at
  all;
* the two `ask_gpt` patches, exercised by *running* the patched upstream file
  against a fake OpenAI SDK -- so a broken patch fails here, not on Kaggle.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VL_ROOT = os.environ.get("ASMRDUB_VL_ROOT", "")


def _load_svc():
    path = os.path.join(ROOT, "runtime", "ollama_svc.py")
    spec = importlib.util.spec_from_file_location("asmrdub_ollama_thinking", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def svc():
    return _load_svc()


# --------------------------------------------------------------------------
# answer_text: where did the server put the answer?
# --------------------------------------------------------------------------


def test_prefers_content_when_present(svc):
    text, field = svc.answer_text({"content": "Canberra", "reasoning": "thinking..."})
    assert (text, field) == ("Canberra", "content")


def test_falls_back_to_reasoning(svc):
    """The exact shape ollama returns for gemma4 via /v1 (issue #15288)."""
    text, field = svc.answer_text({"role": "assistant", "content": "", "reasoning": "堪培拉"})
    assert (text, field) == ("堪培拉", "reasoning")


def test_falls_back_to_reasoning_content(svc):
    """Some servers spell it reasoning_content (deepseek-style)."""
    text, field = svc.answer_text({"content": None, "reasoning_content": "答案"})
    assert (text, field) == ("答案", "reasoning_content")


def test_falls_back_to_thinking(svc):
    text, field = svc.answer_text({"content": "   ", "thinking": "答案"})
    assert (text, field) == ("答案", "thinking")


def test_whitespace_only_content_is_not_an_answer(svc):
    text, field = svc.answer_text({"content": "  \n ", "reasoning": "real"})
    assert (text, field) == ("real", "reasoning")


def test_no_text_anywhere_returns_empty(svc):
    assert svc.answer_text({"role": "assistant", "content": ""}) == ("", "")


def test_non_string_values_are_ignored(svc):
    """A structured reasoning block must not be mistaken for text."""
    text, field = svc.answer_text(
        {"content": "", "reasoning": [{"type": "thinking"}], "thinking": "答案"}
    )
    assert (text, field) == ("答案", "thinking")


# --------------------------------------------------------------------------
# verify: against a fake ollama that can play each shape
# --------------------------------------------------------------------------


class _Fake(BaseHTTPRequestHandler):
    shape = "content"          # content | reasoning | empty | http500
    seen_requests: list = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen_requests.append(body)

        if type(self).shape == "http500":
            self._json(500, {"error": "unsupported architecture"})
            return

        message = {"role": "assistant"}
        if type(self).shape == "content":
            message["content"] = "你好。"
        elif type(self).shape == "reasoning":
            message["content"] = ""
            message["reasoning"] = "用户要翻译。译文是：你好。"
        else:                     # empty
            message["content"] = ""

        self._json(200, {"choices": [{"index": 0, "message": message}]})

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
def fake(request):
    _Fake.shape = "content"
    _Fake.seen_requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1], _Fake
    httpd.shutdown()
    httpd.server_close()


def test_verify_passes_on_a_normal_model(svc, fake):
    port, handler = fake
    handler.shape = "content"
    assert svc.verify("m", port=port, timeout=15) is True


def test_verify_passes_when_the_answer_is_in_reasoning(svc, fake):
    """This is the case that wrongly failed and aborted the whole run."""
    port, handler = fake
    handler.shape = "reasoning"
    assert svc.verify("m", port=port, timeout=15) is True


def test_verify_fails_when_there_is_no_text_at_all(svc, fake):
    port, handler = fake
    handler.shape = "empty"
    assert svc.verify("m", port=port, timeout=15) is False


def test_verify_fails_on_http_error(svc, fake):
    port, handler = fake
    handler.shape = "http500"
    assert svc.verify("m", port=port, timeout=15) is False


def test_verify_asks_for_thinking_to_be_disabled(svc, fake):
    """Without this the model burns tokens on a thinking pass every call."""
    port, handler = fake
    svc.verify("m", port=port, timeout=15)
    assert handler.seen_requests
    assert handler.seen_requests[-1].get("reasoning_effort") == "none"


def test_verify_log_does_not_claim_ok_on_empty(svc, fake, capsys):
    """`verify ok: ''` was the misleading line; it must not reappear."""
    port, handler = fake
    handler.shape = "empty"
    svc.verify("m", port=port, timeout=15)
    output = capsys.readouterr().out
    assert "verify ok" not in output
    assert "no text in any known field" in output


def test_verify_warns_when_thinking_could_not_be_disabled(svc, fake, capsys):
    port, handler = fake
    handler.shape = "reasoning"
    svc.verify("m", port=port, timeout=15)
    output = capsys.readouterr().out
    assert "WARNING" in output and "reasoning" in output


def test_ensure_model_accepts_a_thinking_model(svc, fake, monkeypatch):
    """End of the chain: a thinking model must be usable, not rejected."""
    port, handler = fake
    handler.shape = "reasoning"
    monkeypatch.setattr(svc, "pull", lambda *a, **k: True)
    monkeypatch.setattr(svc, "list_models", lambda port=None: [])
    chosen = svc.ensure_model("ollama", "/tmp/models", port=port)
    assert chosen == svc.pins.OLLAMA_MODEL


# --------------------------------------------------------------------------
# The ask_gpt patches, executed against a fake OpenAI SDK
# --------------------------------------------------------------------------

videolingo = pytest.mark.skipif(
    not VL_ROOT or not os.path.isfile(os.path.join(VL_ROOT, "core/utils/ask_gpt.py")),
    reason="set ASMRDUB_VL_ROOT to a patched VideoLingo checkout",
)


class _FakeCompletions:
    """Stands in for `client.chat.completions`, recording kwargs."""

    def __init__(self, message, reject_reasoning=None):
        self.message = message
        self.reject_reasoning = reject_reasoning     # None | "type" | "http"
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "reasoning_effort" in kwargs and self.reject_reasoning:
            if self.reject_reasoning == "type":
                raise TypeError("got an unexpected keyword argument 'reasoning_effort'")
            raise RuntimeError(
                "Error code: 400 - unknown field: reasoning_effort"
            )
        message = types.SimpleNamespace(**self.message)
        if not hasattr(message, "model_extra"):
            message.model_extra = {
                k: v for k, v in self.message.items() if k != "content"
            }
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


def _run_ask_gpt(tmp_path, monkeypatch, message, reject_reasoning=None,
                 resp_type=None):
    """Import the patched ask_gpt with a fake `openai` and call it.

    Runs the real patched file, so a syntactically valid but behaviourally wrong
    patch still fails this test.
    """
    completions = _FakeCompletions(message, reject_reasoning)

    fake_openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.chat = types.SimpleNamespace(completions=completions)

    fake_openai.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    monkeypatch.syspath_prepend(VL_ROOT)
    monkeypatch.chdir(tmp_path)
    # ask_gpt writes its cache under the cwd; give it a clean one each time.
    for name in [n for n in sys.modules if n.startswith("core")]:
        del sys.modules[name]

    config_source = os.path.join(VL_ROOT, "config.yaml")
    import shutil

    shutil.copy2(config_source, tmp_path / "config.yaml")

    spec = importlib.util.spec_from_file_location(
        "asmrdub_ask_gpt", os.path.join(VL_ROOT, "core", "utils", "ask_gpt.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.ask_gpt("翻译这句话", resp_type=resp_type, log_title="test")
    return result, completions


@videolingo
def test_patched_ask_gpt_reads_content_normally(tmp_path, monkeypatch):
    result, completions = _run_ask_gpt(
        tmp_path, monkeypatch, {"content": "你好。"}
    )
    assert result == "你好。"
    assert completions.calls[0].get("reasoning_effort") == "none"


@videolingo
def test_patched_ask_gpt_recovers_from_an_empty_content(tmp_path, monkeypatch):
    """The failure that would have wasted the whole translation stage."""
    result, _ = _run_ask_gpt(
        tmp_path, monkeypatch, {"content": "", "reasoning": "你好。"}
    )
    assert result == "你好。"


@videolingo
def test_patched_ask_gpt_reads_model_extra(tmp_path, monkeypatch):
    """Pydantic models put unknown fields in model_extra, not as attributes."""
    completions_message = {"content": ""}

    class MessageWithExtra:
        content = ""
        model_extra = {"reasoning": "你好。"}

    fake_openai = types.ModuleType("openai")
    recorded = {}

    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            def create(**kwargs):
                recorded.update(kwargs)
                choice = types.SimpleNamespace(message=MessageWithExtra())
                return types.SimpleNamespace(choices=[choice])

            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    fake_openai.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.syspath_prepend(VL_ROOT)
    monkeypatch.chdir(tmp_path)
    for name in [n for n in sys.modules if n.startswith("core")]:
        del sys.modules[name]

    import shutil

    shutil.copy2(os.path.join(VL_ROOT, "config.yaml"), tmp_path / "config.yaml")
    spec = importlib.util.spec_from_file_location(
        "asmrdub_ask_gpt_extra", os.path.join(VL_ROOT, "core", "utils", "ask_gpt.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.ask_gpt("翻译", log_title="test_extra") == "你好。"
    del completions_message


@videolingo
def test_patched_ask_gpt_retries_without_the_kwarg_on_typeerror(tmp_path, monkeypatch):
    """An older SDK that has no such parameter must not break the call."""
    result, completions = _run_ask_gpt(
        tmp_path, monkeypatch, {"content": "你好。"}, reject_reasoning="type"
    )
    assert result == "你好。"
    assert len(completions.calls) == 2
    assert "reasoning_effort" in completions.calls[0]
    assert "reasoning_effort" not in completions.calls[1]


@videolingo
def test_patched_ask_gpt_retries_without_the_kwarg_on_400(tmp_path, monkeypatch):
    """A strict OpenAI-compatible server that rejects unknown fields."""
    result, completions = _run_ask_gpt(
        tmp_path, monkeypatch, {"content": "你好。"}, reject_reasoning="http"
    )
    assert result == "你好。"
    assert len(completions.calls) == 2


@videolingo
def test_patched_ask_gpt_still_raises_real_errors(tmp_path, monkeypatch):
    """A genuine failure must not be swallowed by the fallback path."""
    fake_openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            def create(**kwargs):
                raise RuntimeError("Error code: 500 - internal server error")

            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )

    fake_openai.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.syspath_prepend(VL_ROOT)
    monkeypatch.chdir(tmp_path)
    for name in [n for n in sys.modules if n.startswith("core")]:
        del sys.modules[name]

    import shutil

    shutil.copy2(os.path.join(VL_ROOT, "config.yaml"), tmp_path / "config.yaml")
    spec = importlib.util.spec_from_file_location(
        "asmrdub_ask_gpt_err", os.path.join(VL_ROOT, "core", "utils", "ask_gpt.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(Exception):
        module.ask_gpt("翻译", log_title="test_err")


@videolingo
def test_patched_ask_gpt_parses_json_from_reasoning(tmp_path, monkeypatch):
    """Every translation prompt asks for JSON; the fallback must feed the parser."""
    payload = '```json\n{"1": {"origin": "こんにちは", "direct": "你好"}}\n```'
    result, _ = _run_ask_gpt(
        tmp_path, monkeypatch, {"content": "", "reasoning": payload},
        resp_type="json",
    )
    assert result["1"]["direct"] == "你好"
