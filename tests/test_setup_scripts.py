"""Smoke tests over the setup/runtime scripts.

These do not touch the network, a GPU, or a real upstream checkout. They cover
the parts that would otherwise only fail 20 minutes into a Kaggle session:
mount detection, symlink composition, and the syntactic validity of the modules
the bootstrap runs as subprocesses.
"""

from __future__ import annotations

import ast
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from setup import prepare_models  # noqa: E402


# --------------------------------------------------------------------------
# Every script must at least parse; a syntax error here is only discovered
# when the bootstrap shells out to it.
# --------------------------------------------------------------------------

SCRIPTS = (
    "kaggle_bootstrap.py",
    "setup/bootstrap_main.py",
    "setup/apply_overlay.py",
    "setup/prepare_env.py",
    "setup/prepare_models.py",
    "runtime/run_all.py",
    "runtime/ollama_svc.py",
    "runtime/tunnel.py",
    "worker/server.py",
    "overlay/asmr_ui.py",
    "overlay/core/_2_asr.py",
    "overlay/core/_8_2_dub_chunks.py",
    "overlay/core/_9_refer_audio.py",
    "overlay/core/_11_merge_audio.py",
    "overlay/core/asr_backend/demucs_vl.py",
    "overlay/core/asr_backend/worker_client.py",
    "overlay/core/tts_backend/custom_tts.py",
)


@pytest.mark.parametrize("relative", SCRIPTS)
def test_script_parses(relative):
    path = os.path.join(ROOT, relative)
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    ast.parse(source, filename=relative)


def test_no_torch_import_in_app_overlay():
    """The app environment has no torch; an import would crash at startup."""
    app_side = [
        "overlay/asmr_ui.py",
        "overlay/core/_2_asr.py",
        "overlay/core/_8_2_dub_chunks.py",
        "overlay/core/_9_refer_audio.py",
        "overlay/core/_11_merge_audio.py",
        "overlay/core/asr_backend/demucs_vl.py",
        "overlay/core/asr_backend/worker_client.py",
        "overlay/core/tts_backend/custom_tts.py",
    ]
    for relative in app_side:
        with open(os.path.join(ROOT, relative), "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("torch", "demucs", "whisperx")), (
                        f"{relative} imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("torch", "demucs", "whisperx")), (
                    f"{relative} imports from {node.module}"
                )


def test_pure_core_imports_nothing_heavy():
    """asmrdub must stay importable on a bare interpreter."""
    for name in os.listdir(os.path.join(ROOT, "asmrdub")):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(ROOT, "asmrdub", name), "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=name)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                top = module.split(".")[0]
                assert top not in {
                    "torch", "numpy", "pandas", "streamlit", "soundfile", "librosa",
                }, f"asmrdub/{name} imports {module}"


# --------------------------------------------------------------------------
# Mount detection
# --------------------------------------------------------------------------


def _make(tmp_path, relative, content=b"x"):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_finds_indextts_by_marker_files(tmp_path):
    for name in ("gpt.pth", "s2mel.pth", "codec.pth", "config.yaml"):
        _make(tmp_path, f"weights/idx/{name}")
    found = prepare_models.find_indextts_dir(str(tmp_path))
    assert found == str(tmp_path / "weights" / "idx")


def test_ignores_partial_indextts_dir(tmp_path):
    _make(tmp_path, "weights/idx/gpt.pth")
    assert prepare_models.find_indextts_dir(str(tmp_path)) is None


def test_finds_whisper_ct2_dir(tmp_path):
    _make(tmp_path, "asr/fw/model.bin")
    _make(tmp_path, "asr/fw/config.json")
    assert prepare_models.find_whisper_dir(str(tmp_path)) == str(
        tmp_path / "asr" / "fw"
    )


def test_missing_root_is_not_an_error(tmp_path):
    absent = str(tmp_path / "nope")
    assert prepare_models.find_indextts_dir(absent) is None
    assert prepare_models.find_whisper_dir(absent) is None
    assert prepare_models.find_gguf(absent) is None


def test_gguf_prefers_name_hint_over_size(tmp_path):
    _make(tmp_path, "a/huge-llama.gguf", b"x" * 5000)
    _make(tmp_path, "b/gemma-4-12b.gguf", b"x" * 100)
    found = prepare_models.find_gguf(str(tmp_path), hint="gemma")
    assert os.path.basename(found) == "gemma-4-12b.gguf"


def test_gguf_falls_back_to_largest(tmp_path):
    _make(tmp_path, "a/small.gguf", b"x" * 10)
    _make(tmp_path, "b/large.gguf", b"x" * 9000)
    found = prepare_models.find_gguf(str(tmp_path), hint="gemma")
    assert os.path.basename(found) == "large.gguf"


def test_walk_respects_depth_limit(tmp_path):
    deep = tmp_path
    for level in range(10):
        deep = deep / f"L{level}"
    deep.mkdir(parents=True)
    visited = list(prepare_models._walk_dirs(str(tmp_path), max_depth=3))
    assert all(path.count(os.sep) - str(tmp_path).count(os.sep) <= 3 for path in visited)


# --------------------------------------------------------------------------
# link() composition
# --------------------------------------------------------------------------


def test_link_creates_reference(tmp_path):
    source = _make(tmp_path, "src/file.bin", b"data")
    target = tmp_path / "dst" / "file.bin"
    prepare_models.link(str(source), str(target))
    assert target.exists()
    assert target.read_bytes() == b"data"


def test_link_replaces_existing(tmp_path):
    first = _make(tmp_path, "src/a.bin", b"one")
    second = _make(tmp_path, "src/b.bin", b"two")
    target = tmp_path / "dst" / "x.bin"
    prepare_models.link(str(first), str(target))
    prepare_models.link(str(second), str(target))
    assert target.read_bytes() == b"two"


def test_link_replaces_directory(tmp_path):
    source_dir = tmp_path / "src" / "dir"
    source_dir.mkdir(parents=True)
    (source_dir / "inner.txt").write_text("hi", encoding="utf-8")
    target = tmp_path / "dst" / "dir"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("old", encoding="utf-8")
    prepare_models.link(str(source_dir), str(target))
    assert (target / "inner.txt").exists()
    assert not (target / "stale.txt").exists()


def test_download_url_rejects_truncated_file(tmp_path, monkeypatch):
    """A short checkpoint must fail here, not inside torch.load later."""
    import io

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        prepare_models.urllib.request,
        "urlopen",
        lambda *a, **k: FakeResponse(b"short"),
    )
    target = str(tmp_path / "model.pt")
    with pytest.raises(RuntimeError, match="expected 999"):
        prepare_models.download_url("http://example/x", target, expected_size=999)
    assert not os.path.exists(target)
    assert not os.path.exists(target + ".part")


def test_download_url_skips_when_size_matches(tmp_path, monkeypatch):
    target = tmp_path / "model.pt"
    target.write_bytes(b"12345")

    def explode(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(prepare_models.urllib.request, "urlopen", explode)
    assert prepare_models.download_url("http://x", str(target), expected_size=5) == str(
        target
    )


# --------------------------------------------------------------------------
# Bootstrap contract
# --------------------------------------------------------------------------


def test_state_file_keys_match_run_all_expectations():
    """run_all reads these keys; bootstrap_main must write all of them."""
    with open(os.path.join(ROOT, "runtime", "run_all.py"), "r", encoding="utf-8") as fh:
        run_all_source = fh.read()
    with open(
        os.path.join(ROOT, "setup", "bootstrap_main.py"), "r", encoding="utf-8"
    ) as fh:
        bootstrap_source = fh.read()
    for key in (
        "repo_root", "index_tts_root", "app_python", "gpu_python",
        "ollama_binary", "cloudflared_binary", "models", "scratch",
        "input_root", "password",
    ):
        assert f'"{key}"' in run_all_source, f"run_all does not read {key}"
        assert f'"{key}"' in bootstrap_source, f"bootstrap does not write {key}"


def test_bootstrap_declares_the_advertised_step_count():
    with open(
        os.path.join(ROOT, "setup", "bootstrap_main.py"), "r", encoding="utf-8"
    ) as fh:
        source = fh.read()
    declared = int(source.split("total = ", 1)[1].split("\n", 1)[0])
    steps = source.count("step(")
    # one `def step(` plus one call per stage
    assert steps - 1 == declared, f"{steps - 1} step() calls vs total={declared}"


def test_pins_resolve_paths_off_kaggle(tmp_path, monkeypatch):
    monkeypatch.setenv("ASMRDUB_HOME", str(tmp_path))
    monkeypatch.setattr(os.path, "isdir", lambda p: False if p == "/kaggle" else True)
    from asmrdub import pins

    dirs = pins.kaggle_dirs()
    assert dirs["on_kaggle"] is False
    assert str(tmp_path) in dirs["working"]


def test_state_json_is_serialisable():
    """Guards against putting a Popen or Path into the state file."""
    sample = {
        "repo_root": "/x",
        "models": {"indextts": "/y", "whisper": "/z", "hdemucs": "/w",
                   "llm_gguf": None},
        "password": "abc",
    }
    assert json.loads(json.dumps(sample)) == sample
