import textwrap

import pytest

from asmrdub.patches import (
    PATCHES,
    apply_patches,
    filter_requirements,
)
from asmrdub.vl_config import config_overrides, flatten


# --------------------------------------------------------------------------
# requirements filtering
# --------------------------------------------------------------------------


def test_torch_pullers_are_dropped():
    kept = filter_requirements(
        [
            "librosa==0.11.0",
            "whisperx>=3.8.1",
            "pyannote-audio>=4.0.0",
            "pytorch-lightning==2.6.1",
            "lightning==2.6.1",
            "requests==2.32.5",
        ]
    )
    assert kept == ["librosa==0.11.0", "requests==2.32.5"]


def test_comments_and_blanks_are_dropped():
    assert filter_requirements(["# a comment", "", "  ", "requests==2.32.5"]) == [
        "requests==2.32.5"
    ]


def test_name_normalisation_matches_underscores_and_bare_names():
    assert filter_requirements(["pyannote_audio", "whisperx"]) == []


def test_environment_markers_survive():
    kept = filter_requirements(['wetext>=0.0.9; sys_platform != "linux"'])
    assert kept == ['wetext>=0.0.9; sys_platform != "linux"']


# --------------------------------------------------------------------------
# patch application
# --------------------------------------------------------------------------


def _write(tmp_path, rel, text):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_patch_is_applied(tmp_path):
    target = _write(tmp_path, "a/b.py", "before\nANCHOR\nafter\n")
    from asmrdub.patches import Patch

    patch = Patch("a/b.py", "ANCHOR\n", "REPLACED\n", "because")
    applied = apply_patches(str(tmp_path), [patch])
    assert target.read_text(encoding="utf-8") == "before\nREPLACED\nafter\n"
    assert applied == ["a/b.py: because"]


def test_rerun_is_idempotent(tmp_path):
    _write(tmp_path, "a/b.py", "ANCHOR\n")
    from asmrdub.patches import Patch

    patch = Patch("a/b.py", "ANCHOR\n", "REPLACED\n", "because")
    apply_patches(str(tmp_path), [patch])
    again = apply_patches(str(tmp_path), [patch])
    assert "already applied" in again[0]


def test_ambiguous_anchor_fails_loudly(tmp_path):
    """Two matches means upstream moved; guessing would corrupt the tree."""
    _write(tmp_path, "a/b.py", "ANCHOR\nANCHOR\n")
    from asmrdub.patches import Patch

    patch = Patch("a/b.py", "ANCHOR\n", "REPLACED\n", "because")
    with pytest.raises(RuntimeError, match="found 2 times"):
        apply_patches(str(tmp_path), [patch])


def test_missing_anchor_fails_loudly(tmp_path):
    _write(tmp_path, "a/b.py", "nothing here\n")
    from asmrdub.patches import Patch

    patch = Patch("a/b.py", "ANCHOR\n", "REPLACED\n", "because")
    with pytest.raises(RuntimeError, match="found 0 times"):
        apply_patches(str(tmp_path), [patch])


def test_missing_file_fails_loudly(tmp_path):
    from asmrdub.patches import Patch

    patch = Patch("nope.py", "a", "b", "because")
    with pytest.raises(RuntimeError, match="patch target missing"):
        apply_patches(str(tmp_path), [patch])


def test_real_patches_apply_to_upstream_fixtures(tmp_path):
    """Fixtures reproduce the pinned upstream lines each patch anchors on."""
    _write(
        tmp_path,
        "core/tts_backend/estimate_duration.py",
        textwrap.dedent(
            """\
            import syllables
            from pypinyin import pinyin, Style
            from g2p_en import G2p
            from typing import Optional
            import re

            class AdvancedSyllableEstimator:
                def __init__(self):
                    self.g2p_en = G2p()
                    self.duration_params = {'en': 0.225}

                def estimate_duration(self, text: str, lang: Optional[str] = None) -> float:
                    return 0.0
            """
        ),
    )
    _write(
        tmp_path,
        "core/tts_backend/tts_main.py",
        "            elif TTS_METHOD == 'custom_tts':\n"
        "                custom_tts(text, save_as)\n",
    )
    _write(
        tmp_path,
        "core/_10_gen_audio.py",
        '        max_workers = load_key("max_workers") if load_key("tts_method") != "gpt_sovits" else 1\n'
        "    tasks_df['real_dur'] = 0\n",
    )
    _write(
        tmp_path,
        "core/utils/ask_gpt.py",
        textwrap.dedent(
            """\
            def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
                params = dict(
                    model=model,
                    messages=messages,
                    response_format=response_format,
                    timeout=300
                )
                resp_raw = client.chat.completions.create(**params)

                # process and return full result
                resp_content = resp_raw.choices[0].message.content
                return resp_content
            """
        ),
    )

    applied = apply_patches(str(tmp_path), PATCHES)
    assert len(applied) == len(PATCHES)

    estimator = (tmp_path / "core/tts_backend/estimate_duration.py").read_text(
        encoding="utf-8"
    )
    assert "from g2p_en import G2p\n" not in estimator.split("def _load_g2p")[0]
    assert "self._g2p_en = None" in estimator
    assert "def g2p_en(self):" in estimator

    tts_main = (tmp_path / "core/tts_backend/tts_main.py").read_text(encoding="utf-8")
    assert "custom_tts(text, save_as, number, task_df)" in tts_main

    gen_audio = (tmp_path / "core/_10_gen_audio.py").read_text(encoding="utf-8")
    assert '"custom_tts"' in gen_audio
    assert "tasks_df['real_dur'] = 0.0" in gen_audio

    ask_gpt = (tmp_path / "core/utils/ask_gpt.py").read_text(encoding="utf-8")
    assert 'reasoning_effort="none"' in ask_gpt
    assert "reasoning_content" in ask_gpt
    # The patched file must still be valid Python.
    import ast

    ast.parse(ask_gpt)


def test_pandas_is_capped_below_3():
    """Upstream assigns floats into int-initialised columns; pandas 3 refuses."""
    from asmrdub.patches import filter_requirements

    kept = filter_requirements(["pandas>=2.2.3\n", "numpy>=2.0.2\n"])
    assert "pandas>=2.2.3,<3" in kept
    assert not any(line == "pandas>=2.2.3" for line in kept), "the cap was not applied"
    # Exactly one pandas line, or the file reads like a conflict.
    assert sum(1 for line in kept if line.startswith("pandas")) == 1


def test_patched_estimator_is_valid_python(tmp_path):
    """The lazy-property rewrite must still compile."""
    _write(
        tmp_path,
        "core/tts_backend/estimate_duration.py",
        textwrap.dedent(
            """\
            from typing import Optional
            from g2p_en import G2p

            class AdvancedSyllableEstimator:
                def __init__(self):
                    self.g2p_en = G2p()

                def estimate_duration(self, text: str, lang: Optional[str] = None) -> float:
                    return 1.0
            """
        ),
    )
    estimator_patches = [p for p in PATCHES if "estimate_duration" in p.path]
    apply_patches(str(tmp_path), estimator_patches)
    source = (tmp_path / "core/tts_backend/estimate_duration.py").read_text(
        encoding="utf-8"
    )
    compile(source, "estimate_duration.py", "exec")


# --------------------------------------------------------------------------
# VideoLingo config overrides
# --------------------------------------------------------------------------


def test_overrides_point_at_local_llm():
    cfg = config_overrides("m", "http://127.0.0.1:11434/v1")
    assert cfg["api"]["base_url"].endswith("/v1")
    assert cfg["api"]["key"], "ask_gpt refuses to run on an empty key"
    assert cfg["api"]["llm_support_json"] is False


def test_overrides_set_japanese_source_and_custom_tts():
    cfg = config_overrides("m", "u")
    assert cfg["whisper"]["language"] == "ja"
    assert cfg["whisper"]["detected_language"] == "ja"
    assert cfg["tts_method"] == "custom_tts"


def test_overrides_disable_video_work():
    cfg = config_overrides("m", "u")
    assert cfg["burn_subtitles"] is False
    assert cfg["ffmpeg_gpu"] is False


def test_speed_ceiling_is_gentle_for_asmr():
    cfg = config_overrides("m", "u")
    assert cfg["speed_factor"]["max"] <= 1.25
    assert cfg["speed_factor"]["accept"] <= cfg["speed_factor"]["max"]
    assert cfg["speed_factor"]["min"] <= cfg["speed_factor"]["accept"]


def test_short_lines_are_not_force_merged():
    cfg = config_overrides("m", "u")
    assert cfg["min_subtitle_duration"] < 2.5


def test_reflection_pass_off_by_default():
    assert config_overrides("m", "u")["reflect_translate"] is False


def test_flatten_produces_dotted_paths():
    flat = dict(flatten(config_overrides("mymodel", "http://x/v1")))
    assert flat["api.model"] == "mymodel"
    assert flat["speed_factor.max"] == 1.20
    assert flat["whisper.language"] == "ja"
    assert flat["max_workers"] == 2


def test_flatten_has_no_nested_values_left():
    for _, value in flatten(config_overrides("m", "u")):
        assert not isinstance(value, dict)
