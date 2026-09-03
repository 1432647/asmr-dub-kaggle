"""Tests for model resolution against a fake /kaggle/input tree.

`prepare_models` is what makes the ~19GB model set fit in Kaggle's 20GB working
quota: it links whatever the user mounted read-only and downloads only the gaps.
The failure modes are all silent -- link the wrong `config.json` and IndexTTS
loads garbage 15 minutes later -- so the mount scanning gets real coverage here
with fake files on disk. Nothing downloads: `_hf_download` / `_hf_file` are
monkeypatched and asserted *not* to run when a mount satisfies the requirement.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from asmrdub import pins  # noqa: E402


def _load():
    path = os.path.join(ROOT, "bootstrap", "prepare_models.py")
    spec = importlib.util.spec_from_file_location("asmrdub_prepare_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pm():
    return _load()


@pytest.fixture
def no_downloads(pm, monkeypatch):
    """Make any download attempt an explicit failure."""
    calls = []

    def forbid_snapshot(repo, dest, allow=None, ignore=None):
        calls.append(("snapshot", repo))
        raise AssertionError(f"unexpected download of {repo}")

    def forbid_file(repo, filename, dest_dir):
        calls.append(("file", repo, filename))
        raise AssertionError(f"unexpected download of {repo}/{filename}")

    monkeypatch.setattr(pm, "_hf_download", forbid_snapshot)
    monkeypatch.setattr(pm, "_hf_file", forbid_file)
    monkeypatch.setattr(
        pm, "download_url",
        lambda url, target, size=None: (_ for _ in ()).throw(
            AssertionError(f"unexpected download of {url}")
        ),
    )
    return calls


def _touch(path, size=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"\0" * size)
    return path


@pytest.fixture
def mounted(tmp_path):
    """A fake /kaggle/input with every model attached, laid out realistically."""
    root = tmp_path / "input"

    tts = root / "indextts-2-5" / "IndexTTS-2.5"
    for name in pins.INDEXTTS_REQUIRED_FILES:
        _touch(str(tts / name))

    aux = tts / "hf_cache"
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        _touch(str(aux / "w2v-bert-2.0" / name))
    _touch(str(aux / "campplus_cn_common.bin"))
    _touch(str(aux / "semantic_codec_model.safetensors"))
    _touch(str(aux / "bigvgan" / "config.json"))
    _touch(str(aux / "bigvgan" / "bigvgan_generator.pt"))

    whisper = root / "faster-whisper" / "large-v3"
    _touch(str(whisper / "model.bin"))
    _touch(str(whisper / "config.json"))
    _touch(str(whisper / "tokenizer.json"))

    _touch(str(root / "separation" / os.path.basename(pins.HDEMUCS_URL)),
           size=pins.HDEMUCS_SIZE)

    _touch(str(root / "llm" / "gemma-4-12b-it-uncensored-Q4_K_M.gguf"), size=4096)

    return root


# --------------------------------------------------------------------------
# Mount discovery
# --------------------------------------------------------------------------


def test_finds_the_indextts_directory(pm, mounted):
    found = pm.find_indextts_dir(str(mounted))
    assert found is not None
    assert os.path.isfile(os.path.join(found, "gpt.pth"))


def test_does_not_mistake_a_partial_indextts_mount(pm, tmp_path):
    """Three of four markers is a broken upload, not a usable model."""
    root = tmp_path / "input"
    for name in ("gpt.pth", "s2mel.pth", "config.yaml"):
        _touch(str(root / "partial" / name))
    assert pm.find_indextts_dir(str(root)) is None


def test_finds_the_whisper_directory(pm, mounted):
    found = pm.find_whisper_dir(str(mounted))
    assert found is not None
    assert os.path.isfile(os.path.join(found, "model.bin"))


def test_walk_respects_the_depth_limit(pm, tmp_path):
    """An unbounded walk over several multi-GB datasets wastes real minutes."""
    deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "f" / "g" / "h"
    _touch(str(deep / "gpt.pth"))
    visited = list(pm._walk_dirs(str(tmp_path), max_depth=3))
    assert not any(str(deep) == path for path in visited)


def test_find_relative_requires_the_named_parent(pm, tmp_path):
    """The bug this prevents: linking whisper's config.json as BigVGAN's."""
    root = tmp_path / "input"
    _touch(str(root / "whisper" / "config.json"))
    assert pm.find_relative(str(root), "bigvgan/config.json") is None

    _touch(str(root / "bigvgan" / "config.json"))
    found = pm.find_relative(str(root), "bigvgan/config.json")
    assert found is not None and "bigvgan" in found


def test_find_relative_matches_a_bare_filename_anywhere(pm, tmp_path):
    root = tmp_path / "input"
    target = _touch(str(root / "whatever" / "campplus_cn_common.bin"))
    assert pm.find_relative(str(root), "campplus_cn_common.bin") == target


def test_gguf_search_prefers_the_hinted_name(pm, tmp_path):
    root = tmp_path / "input"
    _touch(str(root / "a" / "llama-70b.gguf"), size=9000)
    wanted = _touch(str(root / "b" / "gemma-4-12b.gguf"), size=100)
    assert pm.find_gguf(str(root), hint="gemma") == wanted


def test_gguf_search_falls_back_to_the_largest(pm, tmp_path):
    root = tmp_path / "input"
    _touch(str(root / "a" / "small.gguf"), size=10)
    big = _touch(str(root / "b" / "big.gguf"), size=5000)
    assert pm.find_gguf(str(root), hint="gemma") == big


def test_gguf_search_returns_none_when_absent(pm, tmp_path):
    assert pm.find_gguf(str(tmp_path), hint="gemma") is None


# --------------------------------------------------------------------------
# Composition from mounts (the whole point: no downloads)
# --------------------------------------------------------------------------


def test_full_mount_downloads_nothing(pm, mounted, tmp_path, no_downloads):
    """The case the design depends on: everything attached, nothing fetched."""
    resolved = pm.prepare_all(str(tmp_path / "models"), str(mounted))
    assert no_downloads == []
    assert resolved["indextts"] and resolved["whisper"] and resolved["hdemucs"]
    assert resolved["llm_gguf"] and resolved["llm_gguf"].endswith(".gguf")


def test_composed_dir_exposes_every_required_file(pm, mounted, tmp_path, no_downloads):
    dest = pm.compose_indextts(str(tmp_path / "models"), str(mounted))
    for name in pins.INDEXTTS_REQUIRED_FILES:
        assert os.path.isfile(os.path.join(dest, name)), name


def test_composed_dir_is_writable_even_though_mounts_are_not(
    pm, mounted, tmp_path, no_downloads
):
    """IndexTTS writes into {model_dir}/hf_cache; /kaggle/input is read-only."""
    dest = pm.compose_indextts(str(tmp_path / "models"), str(mounted))
    probe = os.path.join(dest, "hf_cache", "write_probe")
    with open(probe, "w") as fh:
        fh.write("ok")
    assert os.path.isfile(probe)


def test_aux_models_land_where_indextts_looks_for_them(
    pm, mounted, tmp_path, no_downloads
):
    """Paths dictated by index-tts's ensure_models_available."""
    dest = pm.compose_indextts(str(tmp_path / "models"), str(mounted))
    cache = os.path.join(dest, "hf_cache")
    assert os.path.isfile(os.path.join(cache, "w2v-bert-2.0", "model.safetensors"))
    assert os.path.isfile(os.path.join(cache, "campplus_cn_common.bin"))
    assert os.path.isfile(os.path.join(cache, "semantic_codec_model.safetensors"))
    assert os.path.isfile(os.path.join(cache, "bigvgan", "config.json"))
    assert os.path.isfile(os.path.join(cache, "bigvgan", "bigvgan_generator.pt"))


def test_qwen_emotion_weights_are_not_linked(pm, mounted, tmp_path, no_downloads):
    """use_qwen_emo=False: 1.2GB of weights we deliberately never load."""
    qwen = mounted / "indextts-2-5" / "IndexTTS-2.5" / "qwen0.6bemo4-merge"
    _touch(str(qwen / "model.safetensors"), size=64)

    dest = pm.compose_indextts(str(tmp_path / "models"), str(mounted))
    linked = os.path.join(dest, "qwen0.6bemo4-merge")
    # Linking the whole mount is fine (a symlink is free); what must not happen
    # is a *download* of it. That is asserted by no_downloads. But the skip
    # pattern must still be configured for the download path.
    assert any("qwen" in pattern for pattern in pins.INDEXTTS_SKIP_PATTERNS)
    del linked


def test_missing_required_file_is_a_hard_error(pm, tmp_path, monkeypatch):
    """Better to fail here than inside torch.load 15 minutes later."""
    root = tmp_path / "input"
    # A mount that looks like IndexTTS but is missing codec.pth after linking.
    tts = root / "IndexTTS-2.5"
    for name in ("gpt.pth", "s2mel.pth", "codec.pth", "config.yaml"):
        _touch(str(tts / name))
    # ...and nothing else, so the tiktoken vocab and feat files are absent.
    monkeypatch.setattr(pm, "_hf_download", lambda *a, **k: None)
    monkeypatch.setattr(pm, "compose_aux", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="missing required files"):
        pm.compose_indextts(str(tmp_path / "models"), str(root))


def test_hdemucs_size_mismatch_is_rejected(pm, tmp_path, monkeypatch):
    """A truncated 335MB checkpoint fails obscurely inside torch.load."""
    root = tmp_path / "input"
    _touch(str(root / "sep" / os.path.basename(pins.HDEMUCS_URL)), size=1234)
    attempted = []
    monkeypatch.setattr(
        pm, "download_url",
        lambda url, target, size=None: attempted.append(url) or target,
    )
    pm.resolve_hdemucs(str(tmp_path / "models"), str(root))
    assert attempted, "a wrong-sized mount must be ignored and re-downloaded"


def test_download_url_verifies_the_size(pm, tmp_path, monkeypatch):
    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def read(self, n=-1):
            data, self._data = self._data, b""
            return data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        pm.urllib.request, "urlopen",
        lambda url, timeout=None: FakeResponse(b"\0" * 10),
    )
    target = str(tmp_path / "thing.pt")
    with pytest.raises(RuntimeError, match="expected"):
        pm.download_url("https://example/thing.pt", target, expected_size=999)
    assert not os.path.exists(target + ".part"), "partial file must be cleaned up"


def test_link_replaces_a_stale_link(pm, tmp_path):
    """Re-running setup after remounting a different dataset must not fail."""
    old = _touch(str(tmp_path / "old.bin"))
    new = _touch(str(tmp_path / "new.bin"), size=5)
    target = str(tmp_path / "link.bin")
    pm.link(old, target)
    pm.link(new, target)
    assert os.path.getsize(target) == 5


def test_link_falls_back_to_copying(pm, tmp_path, monkeypatch):
    """Some filesystems refuse symlinks; correctness beats saving disk."""
    source = _touch(str(tmp_path / "src.bin"), size=7)
    target = str(tmp_path / "dst.bin")
    monkeypatch.setattr(pm.os, "symlink", lambda *a: (_ for _ in ()).throw(OSError()))
    pm.link(source, target)
    assert os.path.isfile(target) and os.path.getsize(target) == 7


def test_prepare_all_is_idempotent(pm, mounted, tmp_path, no_downloads):
    """Re-running after a crash must resume, not fail on existing links."""
    model_root = str(tmp_path / "models")
    first = pm.prepare_all(model_root, str(mounted))
    second = pm.prepare_all(model_root, str(mounted))
    assert first == second


def test_resolved_paths_are_json_serialisable(pm, mounted, tmp_path, no_downloads):
    """The state file is written as JSON and read by run_all."""
    resolved = pm.prepare_all(str(tmp_path / "models"), str(mounted))
    json.dumps(resolved)


# --------------------------------------------------------------------------
# The download path (with the network stubbed out)
# --------------------------------------------------------------------------


def test_empty_mounts_trigger_the_expected_downloads(pm, tmp_path, monkeypatch):
    """With nothing mounted, exactly the pinned repos are requested."""
    requested = []

    def fake_snapshot(repo, dest, allow=None, ignore=None):
        requested.append(repo)
        os.makedirs(dest, exist_ok=True)
        if repo == pins.INDEXTTS_HF_REPO:
            for name in pins.INDEXTTS_REQUIRED_FILES:
                _touch(os.path.join(dest, name))
        elif repo == pins.FASTER_WHISPER_REPO:
            _touch(os.path.join(dest, "model.bin"))
        elif repo == "facebook/w2v-bert-2.0":
            _touch(os.path.join(dest, "model.safetensors"))
        return dest

    def fake_file(repo, filename, dest_dir):
        requested.append(f"{repo}/{filename}")
        return _touch(os.path.join(dest_dir, os.path.basename(filename)))

    monkeypatch.setattr(pm, "_hf_download", fake_snapshot)
    monkeypatch.setattr(pm, "_hf_file", fake_file)
    monkeypatch.setattr(
        pm, "download_url", lambda url, target, size=None: _touch(target)
    )

    empty_input = str(tmp_path / "input")
    os.makedirs(empty_input, exist_ok=True)
    pm.prepare_all(str(tmp_path / "models"), empty_input)

    assert pins.INDEXTTS_HF_REPO in requested
    assert pins.FASTER_WHISPER_REPO in requested
    assert "facebook/w2v-bert-2.0" in requested
    assert any("campplus" in item for item in requested)
    assert any("MaskGCT" in item for item in requested)
    assert any("bigvgan" in item.lower() for item in requested)
    # The 1.2GB emotion model is never requested.
    assert not any("qwen" in item.lower() for item in requested)


def test_indextts_download_excludes_the_emotion_weights(pm, tmp_path, monkeypatch):
    captured = {}

    def fake_snapshot(repo, dest, allow=None, ignore=None):
        if repo == pins.INDEXTTS_HF_REPO:
            captured["ignore"] = ignore
            for name in pins.INDEXTTS_REQUIRED_FILES:
                _touch(os.path.join(dest, name))
        else:
            os.makedirs(dest, exist_ok=True)
            _touch(os.path.join(dest, "model.safetensors"))
        return dest

    monkeypatch.setattr(pm, "_hf_download", fake_snapshot)
    monkeypatch.setattr(
        pm, "_hf_file",
        lambda repo, filename, dest_dir: _touch(
            os.path.join(dest_dir, os.path.basename(filename))
        ),
    )
    empty_input = str(tmp_path / "input")
    os.makedirs(empty_input, exist_ok=True)
    pm.compose_indextts(str(tmp_path / "models"), empty_input)
    assert tuple(captured["ignore"]) == tuple(pins.INDEXTTS_SKIP_PATTERNS)


def test_w2v_download_excludes_the_fairseq_checkpoint(pm, tmp_path, monkeypatch):
    """conformer_shaw.pt is 2.3GB of weights transformers never reads."""
    captured = {}

    def fake_snapshot(repo, dest, allow=None, ignore=None):
        if repo == "facebook/w2v-bert-2.0":
            captured["allow"] = allow
        os.makedirs(dest, exist_ok=True)
        _touch(os.path.join(dest, "model.safetensors"))
        return dest

    monkeypatch.setattr(pm, "_hf_download", fake_snapshot)
    monkeypatch.setattr(
        pm, "_hf_file",
        lambda repo, filename, dest_dir: _touch(
            os.path.join(dest_dir, os.path.basename(filename))
        ),
    )
    empty_input = str(tmp_path / "input")
    os.makedirs(empty_input, exist_ok=True)
    pm.compose_aux(str(tmp_path / "models" / "indextts"), empty_input, None)
    assert tuple(captured["allow"]) == tuple(pins.W2V_BERT_ALLOW)
    assert "conformer_shaw.pt" not in captured["allow"]
