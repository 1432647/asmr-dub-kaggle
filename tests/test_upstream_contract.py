"""Contract tests against the pinned upstream sources.

The worker and the overlay call into VideoLingo and IndexTTS by keyword. A
renamed parameter upstream would not fail here at import time -- it would fail
on Kaggle, after the models have loaded, minutes into a run. So these tests
parse the pinned checkouts with `ast` (no imports, no torch, no GPU) and assert
that every keyword we pass actually exists.

Skipped unless the checkouts are present:
    ASMRDUB_VL_ROOT  -- VideoLingo at the pinned commit
    ASMRDUB_IT_ROOT  -- index-tts at the pinned commit
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VL_ROOT = os.environ.get("ASMRDUB_VL_ROOT", "")
IT_ROOT = os.environ.get("ASMRDUB_IT_ROOT", "")


def _params(path: str, func: str, cls: str | None = None) -> set[str]:
    """Every accepted keyword of a function, including **kwargs presence."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    scope = tree
    if cls is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls:
                scope = node
                break
        else:
            pytest.fail(f"class {cls} not found in {path}")

    for node in ast.walk(scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            args = node.args
            names = {a.arg for a in args.args} | {a.arg for a in args.kwonlyargs}
            names |= {a.arg for a in args.posonlyargs}
            if args.kwarg is not None:
                names.add("**")
            return names
    pytest.fail(f"function {func} not found in {path}")


# --------------------------------------------------------------------------
# IndexTTS
# --------------------------------------------------------------------------

indextts = pytest.mark.skipif(
    not IT_ROOT or not os.path.isfile(os.path.join(IT_ROOT, "indextts", "infer_v2_5.py")),
    reason="set ASMRDUB_IT_ROOT to a pinned index-tts checkout",
)


@indextts
def test_indextts_constructor_kwargs_exist():
    """Exactly the keywords worker.ensure_tts passes."""
    path = os.path.join(IT_ROOT, "indextts", "infer_v2_5.py")
    accepted = _params(path, "__init__", cls="IndexTTS2")
    used = {
        "cfg_path", "model_dir", "use_bf16", "device", "use_cuda_kernel",
        "use_deepspeed", "use_accel", "use_torch_compile", "use_qwen_emo",
    }
    assert used <= accepted, f"unknown IndexTTS2 kwargs: {sorted(used - accepted)}"


@indextts
def test_indextts_infer_kwargs_exist():
    path = os.path.join(IT_ROOT, "indextts", "infer_v2_5.py")
    accepted = _params(path, "infer", cls="IndexTTS2")
    used = {
        "spk_audio_prompt", "text", "output_path", "lang", "verbose",
        "max_text_tokens_per_segment", "interval_silence", "duration_factor",
    }
    assert used <= accepted, f"unknown infer kwargs: {sorted(used - accepted)}"


@indextts
def test_emotion_falls_back_to_the_speaker_prompt():
    """Our whole emotion strategy rests on this: no emo prompt -> use the speaker's.

    If upstream ever stops defaulting `emo_audio_prompt` to `spk_audio_prompt`,
    every line would be synthesised flat and we would need to pass the reference
    twice instead.
    """
    path = os.path.join(IT_ROOT, "indextts", "infer_v2_5.py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    assert "emo_audio_prompt = spk_audio_prompt" in source.replace(" ", " ")


@indextts
def test_hf_cache_is_still_relative_to_the_cwd():
    """Justifies the worker's chdir + checkpoints symlink."""
    path = os.path.join(IT_ROOT, "indextts", "infer_v2_5.py")
    with open(path, encoding="utf-8") as fh:
        head = fh.read(2000)
    assert "HF_HUB_CACHE" in head
    assert "./checkpoints/hf_cache" in head


@indextts
def test_aux_model_destinations_match_upstreams_expectations():
    """pins.AUX_MODELS must land files exactly where infer_v2_5 looks for them."""
    from asmrdub import pins

    path = os.path.join(IT_ROOT, "indextts", "infer_v2_5.py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()

    destinations = {dest for _, _, _, dest in pins.AUX_MODELS}
    # Upstream reads these from {model_dir}/hf_cache/<name>.
    for expected in ("w2v-bert-2.0", "campplus_cn_common.bin", "bigvgan"):
        assert f'"hf_cache", "{expected}"' in source or \
               f"'hf_cache', '{expected}'" in source, expected
        assert any(dest.split("/")[0] == expected or dest == expected
                   for dest in destinations), f"pins is missing {expected}"


@indextts
def test_low_vram_chunking_will_not_trigger_on_a_t4():
    """T4 is 15GB; the <10GB path would silently re-split our text."""
    path = os.path.join(IT_ROOT, "indextts", "infer_v2_5.py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    assert "total_vram_gb < 10.0" in source


@indextts
def test_zh_is_a_valid_language_token():
    """`lang="ZH"` must resolve, or every line silently falls back to 'common'."""
    path = os.path.join(IT_ROOT, "indextts", "utils", "tokenizer.py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    assert '"zh":' in source
    assert "LANGUAGE_DICT" in source


@indextts
def test_python_requirement_still_matches_our_venv_pin():
    """We build the worker venv with -p 3.11; upstream requires >=3.10,<3.12."""
    path = os.path.join(IT_ROOT, "pyproject.toml")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    assert 'requires-python = ">=3.10,<3.12"' in source


@indextts
def test_flash_attn_is_still_an_optional_extra():
    """It cannot run on Turing, so `uv sync` must not pull it by default."""
    path = os.path.join(IT_ROOT, "pyproject.toml")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    optional = source.split("[project.optional-dependencies]", 1)
    assert len(optional) == 2, "no optional-dependencies section"
    base = source.split("[project.optional-dependencies]", 1)[0]
    assert "flash-attn" not in base, "flash-attn became a hard dependency"
    assert "flash-attn" in optional[1]


# --------------------------------------------------------------------------
# VideoLingo
# --------------------------------------------------------------------------

videolingo = pytest.mark.skipif(
    not VL_ROOT or not os.path.isdir(VL_ROOT),
    reason="set ASMRDUB_VL_ROOT to a patched VideoLingo checkout",
)


@videolingo
def test_gen_audio_still_calls_tts_main_per_line():
    """Our custom_tts signature patch depends on this call shape."""
    path = os.path.join(VL_ROOT, "core", "_10_gen_audio.py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    assert "tts_main(" in source
    assert "number" in source


@videolingo
def test_tts_main_dispatches_to_custom_tts():
    path = os.path.join(VL_ROOT, "core", "tts_backend", "tts_main.py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    assert "custom_tts" in source


@videolingo
def test_audio_only_detection_helper_still_exists():
    """The pipeline relies on upstream skipping every video step for audio input."""
    from pathlib import Path

    hits = list(Path(VL_ROOT, "core").rglob("*.py"))
    assert any(
        "is_audio_only_input" in path.read_text(encoding="utf-8", errors="ignore")
        for path in hits
    )


@videolingo
def test_config_keys_we_override_all_exist():
    """A typo'd key writes a new one and leaves upstream's default in force."""
    import ruamel.yaml

    from asmrdub.vl_config import config_overrides, flatten

    with open(os.path.join(VL_ROOT, "config.yaml"), encoding="utf-8") as fh:
        config = ruamel.yaml.YAML().load(fh)

    existing = {key for key, _ in flatten(config)}
    ours = {key for key, _ in flatten(config_overrides("dummy-model", "http://x/v1"))}
    unknown = ours - existing
    assert not unknown, f"config keys not present upstream: {sorted(unknown)}"
