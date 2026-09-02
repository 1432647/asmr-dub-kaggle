"""Tests for pin values and the disk-space-aware scratch selection.

`pins.py` is data, but two things in it are logic worth testing: the scratch
picker (getting it wrong means running out of disk 20 minutes into a Kaggle
session) and the internal consistency of the model lists that
`prepare_models.py` iterates over.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from asmrdub import pins  # noqa: E402


# --------------------------------------------------------------------------
# Scratch selection
# --------------------------------------------------------------------------


def test_prefers_the_first_candidate_with_enough_room(monkeypatch, tmp_path):
    """Order is meaningful: /kaggle/temp beats /tmp even if /tmp is bigger."""
    first = tmp_path / "temp"
    second = tmp_path / "tmp"
    for path in (first, second):
        path.mkdir()

    sizes = {str(first): 40.0, str(second): 900.0}
    monkeypatch.setattr(pins, "free_gb", lambda path: sizes[path])
    chosen = pins.pick_scratch([str(first), str(second)], need_gb=30.0)
    assert chosen == str(first)


def test_skips_a_candidate_that_is_too_small(monkeypatch, tmp_path):
    small = tmp_path / "small"
    big = tmp_path / "big"
    for path in (small, big):
        path.mkdir()

    sizes = {str(small): 5.0, str(big): 60.0}
    monkeypatch.setattr(pins, "free_gb", lambda path: sizes[path])
    chosen = pins.pick_scratch([str(small), str(big)], need_gb=30.0)
    assert chosen == str(big)


def test_falls_back_to_the_roomiest_when_none_are_big_enough(monkeypatch, tmp_path):
    """Better to get partway and hit a real ENOSPC than to pick the smallest."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for path in (a, b):
        path.mkdir()

    sizes = {str(a): 8.0, str(b): 19.0}
    monkeypatch.setattr(pins, "free_gb", lambda path: sizes[path])
    assert pins.pick_scratch([str(a), str(b)], need_gb=30.0) == str(b)


def test_skips_unwritable_candidates(monkeypatch, tmp_path):
    """/kaggle/temp does not exist off-Kaggle; it must not be selected."""
    real = tmp_path / "real"
    real.mkdir()
    missing = tmp_path / "nope" / "deeper"

    monkeypatch.setattr(pins, "_writable", lambda path: path == str(real))
    monkeypatch.setattr(pins, "free_gb", lambda path: 100.0)
    assert pins.pick_scratch([str(missing), str(real)]) == str(real)


def test_returns_last_candidate_when_nothing_is_writable(monkeypatch):
    monkeypatch.setattr(pins, "_writable", lambda path: False)
    assert pins.pick_scratch(["/a", "/b", "/c"]) == "/c"


def test_free_gb_reports_a_real_number_for_a_real_path(tmp_path):
    free = pins.free_gb(str(tmp_path))
    assert free > 0, "a real directory should report positive free space"


def test_free_gb_does_not_raise_on_a_missing_path():
    assert pins.free_gb("/definitely/not/here/at/all") == -1.0


def test_scratch_override_is_honoured(monkeypatch, tmp_path):
    """ASMRDUB_SCRATCH lets a user redirect 30GB without editing code."""
    override = tmp_path / "elsewhere"
    override.mkdir()
    monkeypatch.setenv("ASMRDUB_SCRATCH", str(override))
    monkeypatch.setenv("ASMRDUB_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(os.path, "isdir", lambda path: (
        False if path == "/kaggle" else os.path.exists(path)
    ))
    dirs = pins.kaggle_dirs()
    assert dirs["scratch"] == str(override)


def test_local_dirs_do_not_point_at_kaggle(monkeypatch, tmp_path):
    monkeypatch.delenv("ASMRDUB_SCRATCH", raising=False)
    monkeypatch.setenv("ASMRDUB_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(os.path, "isdir", lambda path: (
        False if path == "/kaggle" else os.path.exists(path)
    ))
    dirs = pins.kaggle_dirs()
    assert dirs["on_kaggle"] is False
    assert "/kaggle" not in dirs["working"]
    assert "/kaggle" not in dirs["input"]


# --------------------------------------------------------------------------
# Pin consistency
# --------------------------------------------------------------------------


def test_upstream_commits_are_full_sha1s():
    """Short SHAs and branch names both silently defeat the point of pinning."""
    for commit in (pins.VIDEOLINGO_COMMIT, pins.INDEXTTS_COMMIT):
        assert len(commit) == 40, commit
        int(commit, 16)  # raises if not hex


def test_aux_models_have_a_known_kind():
    for kind, repo, remote, dest in pins.AUX_MODELS:
        assert kind in ("dir", "file"), kind
        assert repo and dest
        if kind == "file":
            assert remote, f"{repo} needs a remote path"


def test_aux_model_destinations_are_unique():
    """A duplicate destination would silently overwrite another model."""
    destinations = [dest for _, _, _, dest in pins.AUX_MODELS]
    assert len(destinations) == len(set(destinations))


def test_w2v_bert_allowlist_excludes_the_fairseq_checkpoint():
    """conformer_shaw.pt is 2.3GB and transformers never reads it."""
    assert "conformer_shaw.pt" not in pins.W2V_BERT_ALLOW
    assert "model.safetensors" in pins.W2V_BERT_ALLOW


def test_qwen_emotion_weights_are_skipped():
    """use_qwen_emo=False, so its 1.2GB has no reason to be downloaded."""
    assert any("qwen0.6bemo4" in pattern for pattern in pins.INDEXTTS_SKIP_PATTERNS)


def test_hdemucs_sources_order_matches_the_worker():
    """The worker indexes stems by position; a reordering silently swaps them."""
    assert pins.HDEMUCS_SOURCES == ("drums", "bass", "other", "vocals")
    assert pins.HDEMUCS_SOURCES.index("vocals") == 3


def test_ports_are_distinct():
    ports = (pins.UI_PORT, pins.WORKER_PORT, pins.OLLAMA_PORT)
    assert len(set(ports)) == 3


def test_gpus_are_distinct():
    """The whole point of the split is that TTS and the LLM do not share a card."""
    assert pins.WORKER_GPU != pins.LLM_GPU


def test_reference_window_bounds_are_sane():
    assert pins.REFER_MIN_SEC < pins.REFER_MAX_SEC
    assert pins.REFER_JOIN_GAP > 0


def test_separation_overlap_is_smaller_than_the_chunk():
    assert 0 < pins.SEP_OVERLAP < pins.SEP_CHUNK_SEC / 2


def test_ollama_tarball_is_the_zstd_archive():
    """The .tgz URL 404s now; only .tar.zst is published."""
    assert pins.OLLAMA_TARBALL.endswith(".tar.zst")


def test_translation_model_is_a_hf_gguf_reference():
    """ollama needs the hf.co/ prefix to pull a GGUF from HuggingFace."""
    assert pins.OLLAMA_MODEL.startswith("hf.co/")
    assert ":" in pins.OLLAMA_MODEL.rsplit("/", 1)[-1], "missing quantisation tag"
